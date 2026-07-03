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
    support_verdict: str
    useful: bool
    retries: int
    rewrite_tries: int

class SelfRAGEngine:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = ConversationMemory(user_id, supabase_admin)

    def run(self, question: str, session_id: str, focus_doc: Optional[str] = None) -> RAGResult:
        # 1. Decide if retrieval is needed (with keyword override)
        book_keywords = ['book', 'chapter', 'author', 'page', 'according to', 'in the book', 'by ', 'atomic habits', 'coddling', 'network marketing', 'untruths', 'funnels']
        if any(kw in question.lower() for kw in book_keywords):
            need_retrieval = True
        else:
            need_retrieval = decide_retrieval(question)

        if not need_retrieval:
            answer = generate_from_context(question, "No context provided, use your general knowledge.")
            return RAGResult(answer, [], False, 'N/A', True, 0, 0)

        # 2. Retrieval loop (with query rewriting)
        retrieval_query = question
        rewrite_tries = 0
        max_rewrite = 1

        while rewrite_tries <= max_rewrite:
            filter_ = {"source_file": focus_doc} if focus_doc else {}
            docs = search_pinecone(retrieval_query, top_k=10, filter_=filter_)
            if not docs:
                if rewrite_tries < max_rewrite:
                    retrieval_query = rewrite_query(question, retrieval_query, "")
                    rewrite_tries += 1
                    continue
                else:
                    return RAGResult("No documents found in the knowledge base.", [], True, 'no', False, 0, rewrite_tries)

            # 3. Batch relevance filter
            relevant_indices = filter_relevant_batch(question, docs)
            if not relevant_indices:
                # Fallback: use top 3 docs
                relevant_indices = list(range(min(3, len(docs))))

            relevant_docs = [docs[i] for i in relevant_indices[:5]]

            # 4. Build context robustly
            context_parts = []
            for doc in relevant_docs:
                meta = doc.get('metadata', {})
                # Try different possible text keys
                text = meta.get('text') or meta.get('chunk_text') or meta.get('content')
                if text:
                    context_parts.append(text)
                else:
                    # If no text, include metadata as string
                    context_parts.append(str(meta))

            context = "\n---\n".join(context_parts)

            # If context is still empty, use all docs (ignore relevance)
            if not context:
                all_texts = []
                for doc in docs[:5]:
                    meta = doc.get('metadata', {})
                    text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or str(meta)
                    all_texts.append(text)
                context = "\n---\n".join(all_texts)

            # If context is still empty, give up
            if not context:
                return RAGResult("I found some documents but couldn't extract readable content.", [], True, 'no', False, 0, rewrite_tries)

            # 5. Generate answer
            answer = generate_from_context(question, context)

            # 6. Check support & usefulness
            support_verdict, useful = check_support_and_usefulness(question, answer, context)

            # 7. If not fully supported, try one revision
            revision_retries = 0
            if support_verdict != 'full':
                answer = revise_answer(question, answer, context)
                support_verdict, useful = check_support_and_usefulness(question, answer, context)
                revision_retries = 1

            # 8. If not useful, rewrite query and retry (once)
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

        return RAGResult("I couldn't find a useful answer after trying.", [], True, 'no', False, 0, rewrite_tries)