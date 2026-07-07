from groq import Groq
from config import get_settings
from typing import List, Dict

settings = get_settings()
groq_client = Groq(api_key=settings.GROQ_API_KEY)
MODEL = settings.GROQ_MODEL

# --- Simple greetings (exact matches) ---
GREETINGS = {
    "hi", "hello", "hey", "how are you", "what's up",
    "good morning", "good evening", "good night", "howdy",
    "how are you doing", "what's going on", "yo"
}

def call_groq(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def is_greeting(question: str) -> bool:
    q = question.lower().strip()
    return q in GREETINGS or q.rstrip('?!.') in GREETINGS

def generate_greeting() -> str:
    import random
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! How may I help you?",
        "Good day! How can I be of service?"
    ])

def generate_from_context(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the user's question based solely on the provided context.
If the context does not contain enough information, say "I don't have an answer related to this question."

Context:
{context}

Question: {question}
Answer:"""
    return call_groq(prompt, max_tokens=500, temperature=0.3)

def generate_from_chunks(question: str, chunks_text: str) -> str:
    """
    Used by self-learning: given a set of chunks (from all documents),
    decide if any can answer the question.
    If yes, answer; else return exactly "NOT_RELATED".
    """
    prompt = f"""You are a content analyst.
The user asked: "{question}"

We have the following text snippets from our documents:

{chunks_text}

Does any of these snippets contain information that can answer the question?
- If yes, provide a concise answer based only on the snippets.
- If no, reply with exactly "NOT_RELATED".

Answer:"""
    resp = call_groq(prompt, max_tokens=300, temperature=0.2)
    if "NOT_RELATED" in resp:
        return None
    return resp.strip()