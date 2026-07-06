from groq import Groq
from config import get_settings
from typing import List, Dict
import re

settings = get_settings()
groq_client = Groq(api_key=settings.GROQ_API_KEY)
MODEL = settings.GROQ_MODEL

# --- Simple greetings / chit-chat that never trigger retrieval ---
GREETINGS = {
    "hi", "hello", "hey", "how are you", "what's up", 
    "good morning", "good evening", "good night", "howdy",
    "how are you doing", "what's going on", "yo"
}

def call_groq(prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def decide_retrieval(question: str) -> bool:
    """
    Deterministic decision:
    - Skip retrieval for exact greetings.
    - For everything else, force retrieval.
    """
    q_lower = question.lower().strip()
    # If the question is exactly a greeting or a common short phrase, skip
    if q_lower in GREETINGS or q_lower in [g + '?' for g in GREETINGS]:
        return False
    # For any other question, we retrieve (including "what is chatgpt", "what is atomic habbit", etc.)
    return True

def generate_from_context(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the user's question based solely on the provided context.
If the context does not contain enough information, say "I don't have an answer related to this question."

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