from groq import Groq
from config import get_settings
from typing import List, Dict, Optional
import random
import re

settings = get_settings()
groq_client = Groq(api_key=settings.GROQ_API_KEY)
MODEL = settings.GROQ_MODEL

# --- Expanded greetings ---
GREETINGS = {
    "hi", "hello", "hey", "how are you", "what's up",
    "good morning", "good evening", "good night", "howdy",
    "how are you doing", "what's going on", "yo",
    "hii", "hiii", "heyy", "heya", "hiya"
}

def call_groq(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return ""

def is_greeting(question: str) -> bool:
    """Check if the question is a simple greeting, ignoring punctuation."""
    q = re.sub(r'[^a-zA-Z\s]', '', question).strip().lower()
    # Also check if the question is very short and contains only a greeting word
    words = q.split()
    if len(words) <= 2 and any(w in GREETINGS for w in words):
        return True
    return q in GREETINGS

def generate_greeting() -> str:
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! How may I help you?",
        "Good day! How can I be of service?"
    ])

def generate_from_context(question: str, context: str) -> str:
    if not context:
        return "I don't have an answer related to this question."
    prompt = f"""You are a helpful assistant. Answer the user's question based solely on the provided context.
If the context does not contain enough information, say "I don't have enough information in the provided documents."

Context:
{context}

Question: {question}
Answer:"""
    return call_groq(prompt, max_tokens=500, temperature=0.3) or "I don't have an answer related to this question."

def generate_from_chunks(question: str, chunks_text: str) -> Optional[str]:
    """
    Used by self-learning: given a set of chunks (from all documents),
    decide if any can answer the question.
    If yes, answer; else return None.
    """
    if not chunks_text:
        return None
    prompt = f"""You are a content analyst.
The user asked: "{question}"

We have the following text snippets from our documents:

{chunks_text}

Does any of these snippets contain information that can answer the question?
- If yes, provide a concise answer based only on the snippets.
- If no, reply with exactly "NOT_RELATED".

Answer:"""
    resp = call_groq(prompt, max_tokens=300, temperature=0.2)
    if not resp or "NOT_RELATED" in resp:
        return None
    return resp.strip()

def is_insufficient_answer(answer: str) -> bool:
    """Check if the answer indicates insufficient information."""
    lower = answer.lower()
    phrases = [
        "i don't have enough information",
        "i don't have an answer",
        "no information",
        "not enough information"
    ]
    return any(p in lower for p in phrases)