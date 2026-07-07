from typing import List, Dict, Optional
from dataclasses import dataclass
from vector_store import search_pinecone, get_all_filenames, upsert_qa_pair
from llm import (
    is_greeting, generate_greeting, generate_from_context,
    generate_from_chunks
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
    # self-learning metadata (optional)
    learned: bool = False

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
        # 1. Check greetings
        if is_greeting(question):
            return RAGResult(generate_greeting(), [], False)

        # 2. Detect if a specific document is mentioned
        mentioned_doc = detect_mentioned_document(question)
        if focus_doc is None and mentioned_doc:
            focus_doc = mentioned_doc

        # 3. Retrieve from Pinecone
        filter_ = {"source_file": focus_doc} if focus_doc else {}
        docs = search_pinecone(question, top_k=10, filter_=filter_)

        # 4. If we have docs, build context and generate answer
        if docs:
            context_parts = []
            for doc in docs:
                meta = doc.get('metadata', {})
                text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
                if text:
                    context_parts.append(text)
            context = "\n\n---\n\n".join(context_parts[:5])

            if context:
                answer = generate_from_context(question, context)
                sources = [{'document': d['metadata'].get('source_file', 'unknown'), 'score': d.get('score', 0)}
                           for d in docs[:5]]
                return RAGResult(answer, sources, True)

        # 5. No answer found – attempt self‑learning
        learned_answer = self._self_learn(question)
        if learned_answer:
            # Store the new QA pair for future use
            upsert_qa_pair(question, learned_answer)
            return RAGResult(learned_answer, [], True, learned=True)

        # 6. Final fallback – no answer
        return RAGResult("I don't have an answer related to this question.", [], True)

    def _self_learn(self, question: str) -> Optional[str]:
        """
        Re‑retrieve a broader set of chunks (across all documents),
        then ask the LLM if any can answer the question.
        If yes, return the answer; else None.
        """
        # Retrieve from all documents (no filter) to get a variety of chunks
        docs = search_pinecone(question, top_k=10, filter_=None)
        if not docs:
            return None

        # Build a text with snippets (shortened to save tokens)
        snippets = []
        for i, doc in enumerate(docs[:8]):
            meta = doc.get('metadata', {})
            text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
            if text:
                # Truncate to 500 chars per chunk
                snippets.append(f"[Snippet {i+1} from {meta.get('source_file', 'unknown')}]:\n{text[:500]}...")

        if not snippets:
            return None

        combined = "\n\n".join(snippets)
        answer = generate_from_chunks(question, combined)
        return answer