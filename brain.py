# brain.py – Vercel-Optimized (Groq Embeddings, 1536-dim) - UPDATED
import os
import re
import json
import requests
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import Counter

from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq


class KnowledgeBrain:
    """Universal knowledge engine – Uses Groq for embeddings."""

    def __init__(
        self,
        pdf_directory: str = "./pdfs",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.pdf_directory = Path(pdf_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print("📥 Using Groq for embeddings (text-embedding-3-small, 1536-dim)")

        # Groq setup
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embedding_model = os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small")

        # Pinecone setup
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_host = os.getenv("PINECONE_INDEX_HOST")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(host=self.index_host)

        # Local metadata
        self.metadata_file = "./brain_metadata.json"
        self.documents_metadata = self._load_metadata()
        self.doc_summaries = self._build_document_summaries()

        stats = self.get_stats()
        print(f"✅ Brain ready: {stats['total_documents']} docs, "
              f"{stats['total_chunks']} chunks\n")

    # ------------------------------------------------------------------
    # EMBEDDING METHODS USING GROQ
    # ------------------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Groq API"""
        embeddings = []
        for text in texts:
            try:
                response = self.groq_client.embeddings.create(
                    model=self.embedding_model,
                    input=text[:8000]  # Limit text length
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"⚠️  Embedding error: {e}")
                embeddings.append([0.0] * 1536)  # 1536-dim for Groq
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Get query embedding using Groq API"""
        try:
            response = self.groq_client.embeddings.create(
                model=self.embedding_model,
                input=query[:8000]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️  Query embedding error: {e}")
            return [0.0] * 1536

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    def _load_metadata(self) -> dict:
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents_metadata, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # DOCUMENT SUMMARIES
    # ------------------------------------------------------------------
    def _build_document_summaries(self) -> dict:
        summaries = {}
        for filename in self.documents_metadata:
            chunks = self._get_chunks_for_file(filename, limit=30)
            if not chunks:
                continue
            full_text = " ".join([c.page_content for c in chunks])
            summaries[filename] = {
                "total_chars": len(full_text),
                "main_entities": self._extract_entities(full_text),
                "key_terms": self._extract_key_terms(full_text),
                "document_type": self._detect_document_type(full_text, filename),
            }
        return summaries

    def _get_chunks_for_file(self, filename: str, limit: int = 30) -> List[Document]:
        dummy = [0.0] * 1536  # 1536-dim for Groq
        results = self.index.query(
            vector=dummy, filter={"source_file": filename},
            top_k=limit, include_metadata=True
        )
        docs = []
        for m in results.matches:
            docs.append(Document(page_content=m.metadata.get("text", ""), metadata=m.metadata))
        return docs

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+\s+(?:University|College|Institute)[^\.,]*)',
        ]
        for pattern in patterns:
            entities.extend(re.findall(pattern, text))
        return [e for e, c in Counter(entities).most_common(15)]

    def _extract_key_terms(self, text: str, max_terms: int = 25) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were'}
        filtered = [w for w in words if w not in stop_words]
        return [term for term, c in Counter(filtered).most_common(max_terms)]

    def _detect_document_type(self, text: str, filename: str) -> str:
        text_lower = text.lower()[:3000]
        filename_lower = filename.lower()
        if any(w in filename_lower for w in ['tonsil', 'surgery', 'medical']):
            return "medical"
        if any(w in filename_lower for w in ['cv', 'resume']):
            return "resume/cv"
        if re.search(r'\b(tonsil|tonsillectomy|surgery)', text_lower):
            return "medical"
        if re.search(r'(curriculum\s*vitae|\bcv\b|linkedin)', text_lower):
            return "resume/cv"
        return self._detect_category(filename)

    def _detect_category(self, filename: str) -> str:
        fn = filename.lower()
        cats = {
            "resume": ["cv", "resume"], "guide": ["manual", "guide", "guideline"],
            "academic": ["transcript", "degree", "proforma"], "medical": ["medical", "tonsil"],
        }
        for cat, kws in cats.items():
            if any(k in fn for k in kws):
                return cat
        return "general"

    # ------------------------------------------------------------------
    # INGESTION
    # ------------------------------------------------------------------
    def ingest_all_pdfs(self) -> Dict:
        pdf_files = list(self.pdf_directory.glob("**/*.pdf"))
        if not pdf_files:
            print("❌ No PDFs found!")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}
        print(f"📚 Found {len(pdf_files)} PDFs\n")
        results = {"total": len(pdf_files), "processed": 0, "skipped": 0, "failed": 0}
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                filename = pdf_path.name
                if filename in self.documents_metadata:
                    print(f"[{i}/{len(pdf_files)}] ⏭️  {filename}")
                    results["skipped"] += 1
                    continue
                print(f"[{i}/{len(pdf_files)}] 📄 {filename}...", end=" ", flush=True)
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                if not pages:
                    print("❌ No content")
                    results["failed"] += 1
                    continue
                category = self._detect_category(filename)
                for page_num, page in enumerate(pages):
                    page.metadata.update({"source_file": filename, "category": category, "page_number": page_num + 1})
                chunks = text_splitter.split_documents(pages)
                texts = [chunk.page_content for chunk in chunks]
                embeddings = self.embed_texts(texts)
                vectors = []
                for chunk_idx, chunk in enumerate(chunks):
                    chunk.metadata["text"] = chunk.page_content
                    chunk.metadata["chunk_id"] = f"{filename}_chunk_{chunk_idx}"
                    vectors.append({"id": chunk.metadata["chunk_id"], "values": embeddings[chunk_idx], "metadata": chunk.metadata})
                for j in range(0, len(vectors), 100):
                    self.index.upsert(vectors=vectors[j:j+100])
                self.documents_metadata[filename] = {"pages": len(pages), "chunks": len(chunks), "category": category, "upload_date": datetime.now().isoformat()}
                print(f"✅ {len(pages)}p → {len(chunks)} chunks [{category}]")
                results["processed"] += 1
                if results["processed"] % 10 == 0:
                    self._save_metadata()
            except Exception as e:
                print(f"❌ {str(e)[:80]}")
                results["failed"] += 1
        self._save_metadata()
        self.doc_summaries = self._build_document_summaries()
        return results

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 4, category: str = None) -> List[Document]:
        qe = self.embed_query(query)
        filter_dict = {"category": category} if category and category != "all" else None
        results = self.index.query(vector=qe, filter=filter_dict, top_k=k * 5, include_metadata=True)
        docs = []
        for m in results.matches:
            docs.append(Document(page_content=m.metadata.get("text", ""), metadata=m.metadata))
        return docs

    def intelligent_search(self, query: str, k: int = 4, category: str = None) -> List[Document]:
        candidates = self.search(query, k=k, category=category)
        if len(candidates) <= k:
            return candidates
        qa = self._analyze_query(query)
        scored = []
        for doc in candidates:
            fn = doc.metadata.get('source_file', '')
            di = self.doc_summaries.get(fn, {})
            dt = di.get('document_type', 'unknown')
            fs = self._score_filename_match(qa, fn)
            ts = self._score_topic_match(qa, dt, di)
            cs = self._score_content_match(qa, doc.page_content)
            es = self._score_entity_match(qa, di)
            scored.append((fs + ts + cs + es, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            MIN_SCORE = max(30, scored[0][0] * 0.5)
        else:
            MIN_SCORE = 30
        seen = set()
        result = []
        for score, doc in scored:
            if score < MIN_SCORE:
                continue
            src = doc.metadata.get('source_file', '')
            if src not in seen:
                seen.add(src)
                result.append(doc)
        if not result and scored:
            for score, doc in scored[:2]:
                src = doc.metadata.get('source_file', '')
                if src not in seen:
                    seen.add(src)
                    result.append(doc)
        return result[:k]

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------
    def _analyze_query(self, query: str) -> dict:
        q = re.sub(r'[?.,!;:()"\'*]', ' ', query).strip().lower()
        words = set(q.split())
        stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'what', 'who', 'how', 'why', 'where', 'when', 'can', 'could', 'would', 'should', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with', 'from', 'to', 'in', 'on', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'they', 'him', 'her', 'his', 'its', 'their', 'get', 'got', 'has', 'have', 'about', 'just', 'also', 'very', 'much', 'such', 'only', 'over', 'into', 'after', 'then', 'than', 'check', 'tell', 'show', 'find', 'know'}
        key_terms = words - stop
        qtype = "general"
        if any(w in q for w in ['who', 'he', 'she', 'his', 'her']):
            qtype = "person_query"
        elif any(w in q for w in ['how', 'steps', 'guide', 'procedure']):
            qtype = "how_to_query"
        elif any(w in q for w in ['what', 'define', 'explain']):
            qtype = "definition_query"
        return {"original": query, "words": words, "entities": [], "type": qtype, "key_terms": key_terms}

    def _score_filename_match(self, qa: dict, fn: str) -> float:
        score = 0
        for t in qa.get('key_terms', []):
            if t in fn.lower():
                score += 60
        return score

    def _score_topic_match(self, qa: dict, dt: str, di: dict) -> float:
        qt = qa.get('type', 'general')
        if qt == "person_query" and dt == "resume/cv":
            return 80
        if qt == "person_query" and dt == "medical":
            return -80
        if qt == "how_to_query" and dt in ["guide/manual", "academic_record"]:
            return 80
        return 0

    def _score_content_match(self, qa: dict, content: str) -> float:
        return sum(1 for t in qa.get('key_terms', []) if t in content.lower()) * 3

    def _score_entity_match(self, qa: dict, di: dict) -> float:
        return 0

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        try:
            total = self.index.describe_index_stats().get('total_vector_count', 0)
        except:
            total = 0
        return {"total_documents": len(self.documents_metadata), "total_chunks": total, "total_pages": sum(m.get("pages", 0) for m in self.documents_metadata.values()), "categories": {}}

    def get_categories(self) -> List[str]:
        return sorted(set(m.get("category", "general") for m in self.documents_metadata.values()))

    def get_all_filenames(self) -> List[str]:
        return list(self.documents_metadata.keys())