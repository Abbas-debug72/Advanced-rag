from typing import List, Dict, Optional
from dataclasses import dataclass
from vector_store import search_pinecone, get_all_filenames, documents_metadata
from llm import (
    decide_retrieval, filter_relevant_batch, generate_from_context,
    check_support_and_usefulness, revise_answer, rewrite_query
)
from memory import ConversationMemory
from db import supabase_admin
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    answer: str
    sources: List[Dict]
    retrieval_used: bool
    support_verdict: str
    useful: bool
    retries: int
    rewrite_tries: int

def normalize_filename(filename: str) -> str:
    """Remove extension, replace underscores/dashes with spaces, lower."""
    name = filename.rsplit('.', 1)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    return name.lower()

def detect_mentioned_document(question: str) -> Optional[str]:
    """
    Returns the exact source_file (as in metadata) if one is mentioned in the question,
    otherwise None.
    """
    q_lower = question.lower()
    all_filenames = get_all_filenames()
    for fname in all_filenames:
        # Normalize both the filename and the question parts
        norm_fname = normalize_filename(fname)
        # Check if the normalized filename appears as a whole word or phrase
        # Also check if the original filename (without extension) appears
        if norm_fname in q_lower or fname.lower().replace('.pdf', '') in q_lower:
            return fname
    return None

class SelfRAGEngine:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = ConversationMemory(user_id, supabase_admin)

    def run(self, question: str, session_id: str, focus_doc: Optional[str] = None) -> RAGResult:
        # 1. Decide if retrieval is needed
        need_retrieval = decide_retrieval(question)

        if not need_retrieval:
            answer = generate_from_context(question, "No context provided, use your general knowledge.")
            return RAGResult(answer, [], False, 'N/A', True, 0, 0)

        # 2. Check if a specific document is mentioned in the question
        mentioned_doc = detect_mentioned_document(question)
        # If the user already set a focus via "only use X", that overrides
        if focus_doc is None and mentioned_doc:
            focus_doc = mentioned_doc

        # 3. Retrieval loop (with query rewriting)
        retrieval_query = question
        rewrite_tries = 0
        max_rewrite = 1

        while rewrite_tries <= max_rewrite:
            # Apply focus filter
            filter_ = {"source_file": focus_doc} if focus_doc else {}
            docs = search_pinecone(retrieval_query, top_k=15, filter_=filter_)

            if not docs:
                if rewrite_tries < max_rewrite:
                    retrieval_query = rewrite_query(question, retrieval_query, "")
                    rewrite_tries += 1
                    continue
                else:
                    return RAGResult("No documents found in the knowledge base.", [], True, 'no', False, 0, rewrite_tries)

            # 4. Batch relevance filter (allow up to 7 relevant)
            relevant_indices = filter_relevant_batch(question, docs)
            if not relevant_indices:
                # Fallback: use top 5 docs
                relevant_indices = list(range(min(5, len(docs))))

            relevant_docs = [docs[i] for i in relevant_indices[:7]]

            # 5. Build context from relevant docs
            context_parts = []
            for doc in relevant_docs:
                meta = doc.get('metadata', {})
                text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
                if text:
                    context_parts.append(text)

            context = "\n\n---\n\n".join(context_parts)

            # If context is empty, fallback to all docs (top 5)
            if not context:
                all_texts = []
                for doc in docs[:5]:
                    meta = doc.get('metadata', {})
                    text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
                    if text:
                        all_texts.append(text)
                context = "\n\n---\n\n".join(all_texts)

            if not context:
                return RAGResult("I found some documents but couldn't extract readable content.", [], True, 'no', False, 0, rewrite_tries)

            # 6. Generate answer from context
            answer = generate_from_context(question, context)

            # 7. Check support & usefulness
            support_verdict, useful = check_support_and_usefulness(question, answer, context)

            # 8. If not fully supported, try one revision
            revision_retries = 0
            if support_verdict != 'full':
                answer = revise_answer(question, answer, context)
                support_verdict, useful = check_support_and_usefulness(question, answer, context)
                revision_retries = 1

            # 9. If not useful and we have rewrite budget, retry
            if not useful and rewrite_tries < max_rewrite:
                retrieval_query = rewrite_query(question, retrieval_query, answer)
                rewrite_tries += 1
                continue
            else:
                # Build sources list (all from the correct document)
                sources = [{'document': d.get('metadata', {}).get('source_file', 'unknown'), 'score': d.get('score', 0)}
                           for d in relevant_docs]
                return RAGResult(
                    answer=answer,
                    sources=sources,
                    retrieval_used=True,
                    support_verdict=support_verdict,
                    useful=useful,
                    retries=revision_retries,
                    rewrite_tries=rewrite_tries
                )

        # Final fallback
        return RAGResult("I couldn't find a useful answer after trying.", [], True, 'no', False, 0, rewrite_tries)