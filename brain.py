# brain.py – Pinecone + BGE-small + Cross-encoder + Semantic Chunking
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter

from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class KnowledgeBrain:
    """Universal knowledge engine – Pinecone + smart retrieval upgrades."""

    def __init__(
        self,
        pdf_directory: str = "./pdfs",
        embedding_model: str = None,           # can be set from env
        chunk_size: int = None,                 # ignored by SemanticChunker
        chunk_overlap: int = None,              # ignored by SemanticChunker
    ):
        self.pdf_directory = Path(pdf_directory)

        # ── 1. Embedding model (BGE-small by default) ──
        model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
        print(f"📥 Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'trust_remote_code': True}
        )

        # ── 2. Cross‑encoder (re‑ranker) ──
        rerank_model = os.getenv(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print(f"📥 Loading re‑ranker: {rerank_model}")
        self.reranker = CrossEncoder(rerank_model)

        # ── 3. Pinecone setup (same as before) ──
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_host = os.getenv("PINECONE_INDEX_HOST")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(host=self.index_host)

        # Local metadata storage
        self.metadata_file = "./brain_metadata.json"
        self.documents_metadata = self._load_metadata()

        # Document summaries (for topic scoring, unchanged)
        self.doc_summaries = self._build_document_summaries()

        stats = self.get_stats()
        print(f"✅ Brain ready: {stats['total_documents']} docs, "
              f"{stats['total_chunks']} chunks\n")

    # ------------------------------------------------------------------
    # METADATA (identical to earlier)
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
    # DOCUMENT SUMMARIES (unchanged)
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
        dummy = [0.0] * 384
        results = self.index.query(
            vector=dummy,
            filter={"source_file": filename},
            top_k=limit,
            include_metadata=True
        )
        docs = []
        for m in results.matches:
            docs.append(Document(
                page_content=m.metadata.get("text", ""),
                metadata=m.metadata
            ))
        return docs

    # (entity extraction, key terms, document type detection, category – all unchanged)
    # I'll keep them identical to your existing code for brevity, but assume they exist.
    # ------------------------------------------------------------------
    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+\s+(?:University|College|Institute|Company|Corp|Inc|Ltd|Hospital|Center)[^\.,]*)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        entity_counts = Counter(entities)
        return [e for e, c in entity_counts.most_common(15)]

    def _extract_key_terms(self, text: str, max_terms: int = 25) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were',
            'about', 'which', 'their', 'there', 'would', 'could', 'should',
            'these', 'those', 'what', 'when', 'where', 'over', 'into', 'also',
            'after', 'before', 'between', 'under', 'above', 'each', 'every',
            'other', 'some', 'such', 'only', 'then', 'than', 'just', 'because',
            'through', 'during', 'being', 'having', 'more', 'most', 'very'
        }
        filtered = [w for w in words if w not in stop_words]
        term_counts = Counter(filtered)
        return [term for term, count in term_counts.most_common(max_terms)]

    def _detect_document_type(self, text: str, filename: str) -> str:
        text_lower = text.lower()[:3000]
        filename_lower = filename.lower()
        if any(w in filename_lower for w in ['tonsil', 'surgery', 'medical', 'health', 'clinical', 'patient']):
            return "medical"
        if any(w in filename_lower for w in ['cv', 'resume']):
            return "resume/cv"
        if any(w in filename_lower for w in ['transcript', 'proforma', 'degree', 'guideline']):
            return "academic_record"
        if re.search(r'\b(tonsil|tonsillectomy|surgery|surgical|gland|adenoid|anesthesia|post.op)', text_lower):
            return "medical"
        if re.search(r'(curriculum\s*vitae|\bcv\b|objective|experience|certification|linkedin|github)', text_lower):
            return "resume/cv"
        if re.search(r'(transcript|proforma|deposit\s*slip|clearance\s*certificate|matric)', text_lower):
            return "academic_record"
        if re.search(r'(guideline|procedure|step\s*\d|instruction|manual)', text_lower):
            return "guide/manual"
        return self._detect_category(filename)

    def _detect_category(self, filename: str) -> str:
        filename_lower = filename.lower()
        categories = {
            "resume": ["cv", "resume", "biodata"],
            "technical": ["tech", "code", "programming", "software", "api", "devops", "sqa"],
            "business": ["business", "report", "financial", "annual"],
            "legal": ["legal", "contract", "agreement", "policy"],
            "guide": ["manual", "guide", "tutorial", "guideline"],
            "academic": ["academic", "course", "syllabus", "transcript", "degree", "proforma"],
            "medical": ["medical", "health", "tonsil", "surgery"],
        }
        for cat, keywords in categories.items():
            if any(kw in filename_lower for kw in keywords):
                return cat
        return "general"

    # ------------------------------------------------------------------
    # INGESTION (now uses SemanticChunker)
    # ------------------------------------------------------------------
    def ingest_all_pdfs(self) -> Dict:
        pdf_files = list(self.pdf_directory.glob("**/*.pdf"))
        if not pdf_files:
            print("❌ No PDFs found!")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

        print(f"📚 Found {len(pdf_files)} PDFs\n")
        results = {"total": len(pdf_files), "processed": 0, "skipped": 0, "failed": 0}

        # Semantic chunker – splits at natural topic breaks
        text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90  # higher = fewer splits
        )

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
                    page.metadata.update({
                        "source_file": filename,
                        "category": category,
                        "page_number": page_num + 1,
                    })

                # Use semantic chunker
                chunks = text_splitter.split_documents(pages)

                vectors = []
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = self.embeddings.embed_documents([chunk.page_content])[0]
                    chunk.metadata["text"] = chunk.page_content
                    chunk.metadata["chunk_id"] = f"{filename}_chunk_{chunk_idx}"
                    vectors.append({
                        "id": chunk.metadata["chunk_id"],
                        "values": embedding,
                        "metadata": chunk.metadata
                    })

                # Upsert in batches
                BATCH = 100
                for j in range(0, len(vectors), BATCH):
                    self.index.upsert(vectors=vectors[j:j+BATCH])

                self.documents_metadata[filename] = {
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "category": category,
                    "upload_date": datetime.now().isoformat()
                }

                print(f"✅ {len(pages)}p → {len(chunks)} semantic chunks [{category}]")
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
    # SEARCH (Pinecone + cross-encoder reranking)
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 4, category: str = None) -> List[Document]:
        """Raw vector search (no reranking yet)."""
        query_embedding = self.embeddings.embed_query(query)
        filter_dict = None
        if category and category != "all":
            filter_dict = {"category": category}

        results = self.index.query(
            vector=query_embedding,
            filter=filter_dict,
            top_k=k * 5,
            include_metadata=True
        )
        docs = []
        for match in results.matches:
            docs.append(Document(
                page_content=match.metadata.get("text", ""),
                metadata=match.metadata
            ))
        return docs

    def _rerank_with_cross_encoder(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """Cross-encoder reranker – gives true relevance scores."""
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def intelligent_search(self, query: str, k: int = 4, category: str = None) -> List[Document]:
        """
        Full intelligent pipeline:
        1. Dense retrieval from Pinecone (top 20)
        2. Cross-encoder reranking (top 10)
        3. Custom topic/entity scoring (your existing logic) on the reduced set (top 4)
        """
        # 1. Retrieve more candidates
        raw_docs = self.search(query, k=5, category=category)  # 5*5=25

        # 2. Rerank with cross-encoder to top 10
        reranked = self._rerank_with_cross_encoder(query, raw_docs, top_k=10)

        # 3. Apply your custom scoring on the remaining candidates
        query_analysis = self._analyze_query(query)
        scored = []
        for doc in reranked:
            filename = doc.metadata.get('source_file', '')
            doc_info = self.doc_summaries.get(filename, {})
            doc_type = doc_info.get('document_type', 'unknown')
            fs = self._score_filename_match(query_analysis, filename)
            ts = self._score_topic_match(query_analysis, doc_type, doc_info)
            cs = self._score_content_match(query_analysis, doc.page_content)
            es = self._score_entity_match(query_analysis, doc_info)
            total = fs + ts + cs + es
            scored.append((total, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Dynamic threshold (unchanged)
        if scored:
            top_score = scored[0][0]
            MIN_SCORE = max(30, top_score * 0.5)
        else:
            MIN_SCORE = 30

        print(f"   🔍 '{query}' [{query_analysis.get('type', '?')}]")
        print(f"   📊 Threshold: {MIN_SCORE:.0f} (top: {scored[0][0]:.0f})")
        for score, doc in scored[:6]:
            fname = doc.metadata.get('source_file', '?')[:45]
            flag = "✅" if score >= MIN_SCORE else "⏭️"
            print(f"     {score:5.0f} {flag} {fname}")

        seen = set()
        result = []
        for score, doc in scored:
            source = doc.metadata.get('source_file', '')
            if score < MIN_SCORE:
                continue
            if source not in seen:
                seen.add(source)
                result.append(doc)

        if not result and scored:
            print("   ⚠️  Fallback to top 2")
            for score, doc in scored[:2]:
                source = doc.metadata.get('source_file', '')
                if source not in seen:
                    seen.add(source)
                    result.append(doc)

        print(f"   ✅ Returning {len(result)} documents")
        return result[:k]

    # Scoring methods (identical to your current code)
    def _analyze_query(self, query: str) -> dict:
        query_clean = re.sub(r'[?.,!;:()"\'*]', ' ', query)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()
        query_lower = query_clean.lower()
        query_words = set(query_lower.split())
        analysis = {
            "original": query,
            "words": query_words,
            "entities": [],
            "type": "general",
            "key_terms": set(),
        }
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', query_clean)
        analysis["entities"] = [n.lower() for n in names]
        if any(w in query_lower for w in ['who', 'whose', 'person', 'he', 'she', 'his', 'her']):
            analysis["type"] = "person_query"
        elif any(w in query_lower for w in ['how', 'procedure', 'process', 'steps', 'guide']):
            analysis["type"] = "how_to_query"
        elif any(w in query_lower for w in ['what', 'define', 'explain', 'meaning']):
            analysis["type"] = "definition_query"
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
            'did', 'has', 'have', 'he', 'she', 'it', 'they', 'him', 'her',
            'his', 'what', 'who', 'where', 'when', 'why', 'how', 'check',
            'tell', 'show', 'find', 'know', 'about', 'me', 'you', 'can',
            'could', 'would', 'should', 'and', 'or', 'but', 'if', 'of',
            'at', 'by', 'for', 'with', 'from', 'to', 'in', 'on', 'this',
            'get', 'got', 'does', 'any', 'some', 'the', 'i', 'my', 'mine',
            'that', 'these', 'those', 'then', 'than', 'just', 'also',
            'very', 'much', 'such', 'only', 'over', 'into', 'after'
        }
        analysis["key_terms"] = {w for w in query_words if w not in stop_words and len(w) > 1}
        return analysis

    def _score_filename_match(self, qa: dict, filename: str) -> float:
        fn = filename.lower()
        score = 0
        for entity in qa.get('entities', []):
            if entity in fn:
                score += 80
        for term in qa.get('key_terms', []):
            if term in fn:
                score += 60
        return score

    def _score_topic_match(self, qa: dict, doc_type: str, doc_info: dict) -> float:
        query_type = qa.get('type', 'general')
        query_terms = qa.get('key_terms', set())
        score = 0
        doc_key_terms = set(doc_info.get('key_terms', []))
        term_overlap = len(query_terms & doc_key_terms)
        if query_type == "person_query":
            if doc_type == "resume/cv":
                score += 80
            elif doc_type == "medical":
                score -= 80
            elif doc_type in ["guide/manual", "academic_record"]:
                score -= 50
        elif query_type == "how_to_query":
            if doc_type in ["guide/manual", "academic_record"]:
                score += 80
            elif doc_type == "resume/cv":
                score -= 60
            elif doc_type == "medical":
                score -= 40
        elif query_type == "definition_query":
            if term_overlap >= 3:
                score += 80
            elif term_overlap >= 1:
                score += 40
            else:
                score -= 20
            medical_terms = {'medical', 'surgery', 'disease', 'tonsil', 'tonsillectomy',
                             'treatment', 'diagnosis', 'clinical', 'patient', 'health'}
            cv_terms = {'cv', 'resume', 'job', 'work', 'experience', 'skills'}
            academic_terms = {'transcript', 'degree', 'form', 'proforma', 'guideline', 'issuance'}
            if query_terms & medical_terms and doc_type == 'medical':
                score += 50
            if query_terms & cv_terms and doc_type == 'resume/cv':
                score += 50
            if query_terms & academic_terms and doc_type in ['academic_record', 'guide/manual']:
                score += 50
        return score

    def _score_content_match(self, qa: dict, content: str) -> float:
        content_lower = content.lower()
        score = 0
        for entity in qa.get('entities', []):
            if entity in content_lower:
                score += 20
        term_count = sum(1 for t in qa.get('key_terms', []) if t in content_lower)
        score += term_count * 3
        return score

    def _score_entity_match(self, qa: dict, doc_info: dict) -> float:
        if not doc_info:
            return 0
        score = 0
        doc_entities = [e.lower() for e in doc_info.get('main_entities', [])]
        for entity in qa.get('entities', []):
            if any(entity in de for de in doc_entities):
                score += 30
        for term in qa.get('key_terms', []):
            if any(term in de for de in doc_entities):
                score += 25
        return score

    # ------------------------------------------------------------------
    # UTILITIES (unchanged)
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        try:
            stats = self.index.describe_index_stats()
            total_chunks = stats.get('total_vector_count', 0)
        except:
            total_chunks = 0
        cats = {}
        for m in self.documents_metadata.values():
            cat = m.get("category", "general")
            cats[cat] = cats.get(cat, 0) + 1
        return {
            "total_documents": len(self.documents_metadata),
            "total_chunks": total_chunks,
            "total_pages": sum(m.get("pages", 0) for m in self.documents_metadata.values()),
            "categories": cats,
        }

    def get_categories(self) -> List[str]:
        return sorted(set(m.get("category", "general") for m in self.documents_metadata.values()))

    def get_all_filenames(self) -> List[str]:
        return list(self.documents_metadata.keys())