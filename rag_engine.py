from typing import List, Dict, Optional
from dataclasses import dataclass
from vector_store import search_pinecone, get_all_filenames
from rerank import rerank_documents
from llm import (
    decide_retrieval, generate_from_context,
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
    name = filename.rsplit('.', 1)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    return name.lower()

def detect_mentioned_document(question: str) -> Optional[str]:
    q_lower = question.lower()
    all_filenames = get_all_filenames()
    for fname in all_filenames:
        norm_fname = normalize_filename(fname)
        if norm_fname in q_lower or fname.lower().replace('.pdf', '') in q_lower:
            return fname
    return None

class SelfRAGEngine:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = ConversationMemory(user_id, supabase_admin)

    def run(self, question: str, session_id: str, focus_doc: Optional[str] = None) -> RAGResult:
        need_retrieval = decide_retrieval(question)

        if not need_retrieval:
            # For greetings, we generate a friendly response without context
            answer = generate_from_context(question, "No context provided, use your general knowledge.")
            return RAGResult(answer, [], False, 'N/A', True, 0, 0)

        # Auto-detect focus if a document is mentioned
        mentioned_doc = detect_mentioned_document(question)
        if focus_doc is None and mentioned_doc:
            focus_doc = mentioned_doc

        retrieval_query = question
        rewrite_tries = 0
        max_rewrite = 1

        while rewrite_tries <= max_rewrite:
            filter_ = {"source_file": focus_doc} if focus_doc else {}
            docs = search_pinecone(retrieval_query, top_k=30, filter_=filter_)

            if not docs:
                # No documents found – return "I don't have an answer"
                return RAGResult(
                    "I don't have an answer related to this question.",
                    [], True, 'no', False, 0, rewrite_tries
                )

            # Rerank and take top 12
            relevant_indices = rerank_documents(question, docs, top_k=12)
            relevant_docs = [docs[i] for i in relevant_indices]

            # Build context
            context_parts = []
            for doc in relevant_docs:
                meta = doc.get('metadata', {})
                text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
                if text:
                    context_parts.append(text)

            context = "\n\n---\n\n".join(context_parts)

            # If still empty, use fallback
            if not context:
                all_texts = []
                for doc in docs[:7]:
                    meta = doc.get('metadata', {})
                    text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
                    if text:
                        all_texts.append(text)
                context = "\n\n---\n\n".join(all_texts)

            if not context:
                return RAGResult(
                    "I don't have an answer related to this question.",
                    [], True, 'no', False, 0, rewrite_tries
                )

            # Generate answer from context
            answer = generate_from_context(question, context)

            support_verdict, useful = check_support_and_usefulness(question, answer, context)

            # Revision if needed
            revision_retries = 0
            if support_verdict != 'full':
                answer = revise_answer(question, answer, context)
                support_verdict, useful = check_support_and_usefulness(question, answer, context)
                revision_retries = 1

            if not useful and rewrite_tries < max_rewrite:
                retrieval_query = rewrite_query(question, retrieval_query, answer)
                rewrite_tries += 1
                continue
            else:
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

        return RAGResult(
            "I don't have an answer related to this question.",
            [], True, 'no', False, 0, rewrite_tries
        )