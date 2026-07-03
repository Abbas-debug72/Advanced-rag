from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from vector_store import search_pinecone, get_all_filenames
from llm import (
    decide_retrieval, filter_relevant_batch, generate_from_context,
    check_support_and_usefulness, revise_answer, rewrite_query
)
from memory import ConversationMemory
from db import supabase_admin
import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    answer: str
    sources: List[Dict]
    retrieval_used: bool
    support_verdict: str  # 'full', 'partial', 'no'
    useful: bool
    retries: int
    rewrite_tries: int

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

        # 2. Retrieval loop (with possible query rewriting)
        retrieval_query = question
        rewrite_tries = 0
        max_rewrite = 1  # we only allow one rewrite

        while rewrite_tries <= max_rewrite:
            # Apply focus filter if any
            filter_ = {"source_file": focus_doc} if focus_doc else {}
            docs = search_pinecone(retrieval_query, top_k=10, filter_=filter_)
            if not docs:
                if rewrite_tries < max_rewrite:
                    retrieval_query = rewrite_query(question, retrieval_query, "")
                    rewrite_tries += 1
                    continue
                else:
                    return RAGResult("No documents found.", [], True, 'no', False, 0, rewrite_tries)

            # 3. Batch relevance filter (one call for all docs)
            relevant_indices = filter_relevant_batch(question, docs)
            if not relevant_indices:
                if rewrite_tries < max_rewrite:
                    retrieval_query = rewrite_query(question, retrieval_query, "")
                    rewrite_tries += 1
                    continue
                else:
                    return RAGResult("No relevant documents found.", [], True, 'no', False, 0, rewrite_tries)

            relevant_docs = [docs[i] for i in relevant_indices[:5]]  # cap
            context = "\n---\n".join([d['metadata'].get('text', '') for d in relevant_docs])

            # 4. Generate answer from context
            answer = generate_from_context(question, context)

            # 5. Combined support & usefulness check
            support_verdict, useful = check_support_and_usefulness(question, answer, context)

            # 6. If not fully supported, try to revise (once)
            revision_retries = 0
            if support_verdict != 'full':
                answer = revise_answer(question, answer, context)
                support_verdict, useful = check_support_and_usefulness(question, answer, context)
                revision_retries = 1

            # If still not useful and we have rewrite budget left, rewrite and retry
            if not useful and rewrite_tries < max_rewrite:
                retrieval_query = rewrite_query(question, retrieval_query, answer)
                rewrite_tries += 1
                continue
            else:
                # Final answer (even if not useful)
                sources = [{'document': d['metadata'].get('source_file', 'unknown'), 'score': d.get('score', 0)}
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

        # fallback
        return RAGResult("I couldn't find a useful answer.", [], True, 'no', False, 0, rewrite_tries)