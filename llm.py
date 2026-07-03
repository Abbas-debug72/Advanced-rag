from groq import Groq
from config import get_settings
from typing import List, Dict
import re

# Import document metadata for keyword matching
from vector_store import documents_metadata

settings = get_settings()
groq_client = Groq(api_key=settings.GROQ_API_KEY)
MODEL = settings.GROQ_MODEL

# Build a list of document titles and authors to force retrieval
DOCUMENT_KEYWORDS = [
    "atomic habits", "james clear", "coddling", "jonathan haidt", "greg lukianoff",
    "network marketing secrets", "the mountain is you", "brianna wiest",
    "untethered soul", "michael singer", "like a house on fire", "berklee online",
    "dragonlance", "greek basic course", "5zbaneshgh"
]
# Add all filenames from metadata
DOCUMENT_KEYWORDS.extend([f.lower() for f in documents_metadata.keys()])

# Trivial patterns that should never trigger retrieval
TRIVIAL_PATTERNS = [
    "how many seconds", "what is the capital", "who is the president",
    "what is the meaning of", "define", "what are the symptoms of",
    "how does gravity work", "what is the speed of light",
    "how are you", "what is your name", "who are you", "hello", "hi",
    "what is the weather", "what time is it", "today's date",
    "what is the square root", "what is the boiling point",
    "who discovered", "when was", "how old is", "what is the population"
]

def call_groq(prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def decide_retrieval(question: str) -> bool:
    q_lower = question.lower()

    # 1. Force retrieval if question mentions a known document or author
    if any(kw in q_lower for kw in DOCUMENT_KEYWORDS):
        return True

    # 2. Skip retrieval for obvious trivial general‑knowledge questions
    if any(p in q_lower for p in TRIVIAL_PATTERNS):
        return False

    # 3. Use LLM to decide for the rest (conservative)
    prompt = f"""You are a router that decides whether to search internal PDF documents.
Question: "{question}"

Do you need to search the PDF documents to answer this question accurately?
Reply with exactly 'yes' or 'no'.

Rules:
- ONLY say 'yes' if the question asks about specific content from a book, author, or document that is likely in the knowledge base (e.g., "According to Atomic Habits...", "What does the book say about...", "In The Coddling...", "Network Marketing Secrets").
- Say 'no' for general knowledge, definitions, common facts, chit‑chat, or anything that a typical AI would know without documents.
- If the question mentions a specific book title, author, or document name, say 'yes' (already caught above, but just in case).
- Otherwise, say 'no'.

Now respond with only 'yes' or 'no':"""
    ans = call_groq(prompt, max_tokens=5).lower()
    return ans.startswith('y')

def filter_relevant_batch(question: str, docs: List[Dict]) -> List[int]:
    """
    Returns indices of documents that are relevant to the question.
    If parsing fails, returns top min(3, len(docs)) as fallback.
    """
    if not docs:
        return []
    # Build short representation of each doc
    doc_texts = []
    for i, doc in enumerate(docs):
        meta = doc.get('metadata', {})
        text = meta.get('text') or meta.get('chunk_text') or meta.get('content') or ""
        # truncate to save tokens
        text = text[:2000]
        doc_texts.append(f"[{i}] {text[:500]}...")
    combined = "\n".join(doc_texts)
    prompt = f"""Question: {question}

Documents:
{combined}

Which documents contain information useful for answering the question?
Reply with a list of indices, e.g., [0, 2, 5] or [] if none.
Only return the list, nothing else."""
    resp = call_groq(prompt, max_tokens=100)
    try:
        numbers = re.findall(r'\d+', resp)
        indices = [int(n) for n in numbers if int(n) < len(docs)]
        return indices
    except:
        # Fallback: return top min(3, len(docs))
        return list(range(min(3, len(docs))))

def generate_from_context(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the user's question based solely on the provided context.
If the context does not contain enough information, say "I don't have enough information in the provided documents."

Context:
{context}

Question: {question}
Answer:"""
    return call_groq(prompt, max_tokens=500, temperature=0.3)

def check_support_and_usefulness(question: str, answer: str, context: str) -> tuple:
    """
    Returns (support_verdict, useful)
    support_verdict: 'full', 'partial', 'no'
    useful: True/False
    """
    prompt = f"""You are a strict evaluator.
Question: {question}
Answer: {answer}
Context: {context}

1. Is the answer fully supported by the context? (All facts in answer appear in context)
   Reply with 'full', 'partial', or 'no' (one word).

2. Does the answer actually answer the user's question?
   Reply with 'yes' or 'no'.

Format your reply exactly as two lines:
support: <word>
useful: <word>

Example:
support: full
useful: yes
"""
    resp = call_groq(prompt, max_tokens=30)
    support = 'no'
    useful = False
    for line in resp.splitlines():
        if line.startswith('support:'):
            val = line.split(':',1)[1].strip().lower()
            if val in ('full', 'partial', 'no'):
                support = val
        elif line.startswith('useful:'):
            val = line.split(':',1)[1].strip().lower()
            useful = val == 'yes'
    return support, useful

def revise_answer(question: str, answer: str, context: str) -> str:
    prompt = f"""Rewrite the answer to remove any facts not present in the context.
Question: {question}
Original answer: {answer}
Context: {context}
New answer (only from context):"""
    return call_groq(prompt, max_tokens=500, temperature=0.2)

def rewrite_query(question: str, previous_query: str, last_answer: str) -> str:
    prompt = f"""Original question: {question}
Previous retrieval query: {previous_query}
The previous answer was not useful.
Write a new, short retrieval query with key terms to find better documents.
Return only the query, nothing else."""
    return call_groq(prompt, max_tokens=50)