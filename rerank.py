from sentence_transformers import CrossEncoder
from typing import List, Dict

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query: str, docs: List[Dict], top_k: int = 12) -> List[int]:
    if not docs:
        return []
    pairs = []
    for doc in docs:
        meta = doc.get('metadata', {})
        text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
        pairs.append((query, text))
    scores = reranker.predict(pairs)
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]