from pinecone import Pinecone
from config import get_settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
import time
import uuid

settings = get_settings()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(host=settings.PINECONE_INDEX_HOST)

# Load embedding model (small memory footprint)
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    embedding_model = None

def get_embedding(text: str) -> List[float]:
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded")
    if len(text) > 8000:
        text = text[:8000]
    return embedding_model.encode(text).tolist()

def search_pinecone(query: str, top_k: int = 10, filter_: Optional[Dict] = None) -> List[Dict]:
    if not query:
        return []
    try:
        q_emb = get_embedding(query)
        results = pinecone_index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
            filter=filter_
        )
        return results.get('matches', [])
    except Exception as e:
        print(f"Pinecone search error: {e}")
        return []

def get_document_metadata():
    try:
        with open("brain_metadata.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Metadata load error: {e}")
        return {}

documents_metadata = get_document_metadata()

def get_all_filenames() -> List[str]:
    return list(documents_metadata.keys())

def upsert_qa_pair(question: str, answer: str):
    """Store a new Q&A pair in Pinecone for future retrieval."""
    try:
        qa_text = f"Question: {question}\nAnswer: {answer}"
        emb = get_embedding(qa_text)
        doc_id = f"qa_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        pinecone_index.upsert(
            vectors=[(doc_id, emb, {
                "source_file": "self_generated",
                "type": "qa",
                "text": qa_text,
                "question": question,
                "answer": answer
            })]
        )
        print(f"✅ Stored new QA pair: {doc_id}")
    except Exception as e:
        print(f"Failed to store QA pair: {e}")