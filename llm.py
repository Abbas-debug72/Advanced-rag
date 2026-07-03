from groq import Groq
from config import get_settings
from typing import List, Dict
import re

settings = get_settings()
groq_client = Groq(api_key=settings.GROQ_API_KEY)
MODEL = settings.GROQ_MODEL

def call_groq(prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def decide_retrieval(question: str) -> bool:
    prompt = f"""You are a router for a knowledge‑based AI.

Question: "{question}"

Should we search the company's internal PDF documents to answer this question?
Reply with exactly 'yes' or 'no'.

Rules:
- Answer 'yes' if the question asks for any specific fact, name, date, number, definition, concept, or reference that might be in the documents.
- Answer 'yes' if the question mentions any book, author, or specific topic that could be covered in the documents.
- Answer 'yes' unless the question is pure chit‑chat, casual greeting, or a general opinion.
- When in doubt, answer 'yes' – it is safer to retrieve than to miss information.

Now respond with only 'yes' or 'no':"""
    ans = call_groq(prompt, max_tokens=5).lower()
    return ans.startswith('y')

def filter_relevant_batch(question: str, docs: List[Dict]) -> List[int]:
    if not docs:
        return []
    doc_texts = []
    for i, doc in enumerate(docs):
        text = doc.get('metadata', {}).get('text', '')[:2000]
        if not text:
            text = doc.get('metadata', {}).get('chunk_text', '')[:2000]
        if not text:
            text = doc.get('metadata', {}).get('content', '')[:2000]
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
        # Fallback: return top 3
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