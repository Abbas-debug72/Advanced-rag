# app.py – Pinecone RAG Chatbot (with HuggingFace embeddings)
from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import re
from flask import Flask, request, jsonify, render_template, session
from brain import KnowledgeBrain
from memory import ConversationMemory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
app.secret_key = os.urandom(24)

print("🧠 Loading Knowledge Brain with free HuggingFace embeddings...")
brain = KnowledgeBrain(pdf_directory=os.getenv("PDF_DIRECTORY", "./pdfs"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set in .env file")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

memory = ConversationMemory()
session_focus = {}

# Improved prompt
PROMPT = """You are a precise and insightful research assistant.  
Your answers are based **only** on the document context below.

**Rules:**
- If the context contains enough detail, provide a **comprehensive answer** with examples, comparisons, or steps (as appropriate).
- Structure your answer clearly: use **bullet points** for lists, **paragraphs** for explanations.
- Do **not** mention the context itself (e.g., "according to the document").
- If the context lacks the answer, say: "I could not find that information in the knowledge base."
- Match the level of detail to the question: a simple question gets a concise answer; a complex or open‑ended question gets a thorough one.

**Active filter:** {focus_info}

**Context from documents:**
{context}

**Conversation history:**
{chat_history}

**Question:** {question}
**Answer:**"""

QA_PROMPT = ChatPromptTemplate.from_template(PROMPT)

def format_docs(docs):
    parts = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get('source_file', '?')
        if src in seen:
            continue
        seen.add(src)
        page = doc.metadata.get('page_number', '?')
        parts.append(f"[{src} p{page}]\n{doc.page_content[:800]}\n")
    return "\n".join(parts)

# Focus management
def resolve_filename(user_input: str):
    all_files = brain.get_all_filenames()
    for f in all_files:
        if user_input.lower() == f.lower():
            return f
    if not user_input.lower().endswith('.pdf'):
        for f in all_files:
            if f.lower() == user_input.lower() + '.pdf':
                return f
    return None

def detect_focus_command(question: str):
    q = question.lower()
    patterns = [
        r'only\s+use\s+([\w\-.]+(?:\.pdf)?)',
        r'focus\s+on\s+([\w\-.]+(?:\.pdf)?)',
        r'use\s+only\s+([\w\-.]+(?:\.pdf)?)',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            candidate = m.group(1).strip()
            resolved = resolve_filename(candidate)
            if resolved:
                return resolved
            return candidate
    if "clear focus" in q or "remove focus" in q or "no filter" in q:
        return "CLEAR"
    return None

# Routes
@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", session.get("session_id", "default"))
    category = data.get("category", "all")

    if not question:
        return jsonify({"error": "Question required"}), 400

    focus_file = session_focus.get(session_id)
    focus_cmd = detect_focus_command(question)

    if focus_cmd == "CLEAR":
        session_focus.pop(session_id, None)
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", "✅ Document filter cleared.")
        return jsonify({"answer": "✅ Document filter cleared.", "sources": [], "focus": None})

    if focus_cmd is not None:
        all_files = brain.get_all_filenames()
        if focus_cmd in all_files:
            session_focus[session_id] = focus_cmd
            msg = f"✅ Now focusing on **{focus_cmd}** only."
        else:
            msg = f"❌ Document `{focus_cmd}` not found. Available: {', '.join(all_files)}"
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", msg)
        return jsonify({"answer": msg, "sources": [], "focus": session_focus.get(session_id)})

    history = memory.get_history(session_id, last_n=6)
    history_str = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]
    ) if history else "No history"

    if focus_file:
        raw_docs = brain.search(question, k=30, category=category if category != "all" else None)
        docs = [d for d in raw_docs if d.metadata.get('source_file', '').lower() == focus_file.lower()]
        if not docs:
            answer = f"I could not find any relevant information inside **{focus_file}**."
            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "assistant", answer)
            return jsonify({"answer": answer, "sources": [], "focus": focus_file})
        docs = docs[:6]
    else:
        docs = brain.intelligent_search(question, k=4, category=category if category != "all" else None)

    context = format_docs(docs)
    focus_info = f"Currently focused on: {focus_file}. Only use this document." if focus_file else "No active document filter."

    chain = QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": history_str,
        "question": question,
        "focus_info": focus_info
    })

    memory.add_message(session_id, "user", question)
    memory.add_message(session_id, "assistant", answer)

    seen_src = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("source_file", "?")
        if src not in seen_src:
            seen_src.add(src)
            sources.append({"document": src, "page": doc.metadata.get("page_number", "?")})

    return jsonify({"answer": answer, "sources": sources, "focus": focus_file})

@app.route("/api/focus", methods=["POST"])
def set_focus():
    data = request.get_json()
    session_id = data.get("session_id", session.get("session_id", "default"))
    filename = data.get("filename", None)
    if filename:
        session_focus[session_id] = filename
    else:
        session_focus.pop(session_id, None)
    return jsonify({"focus": session_focus.get(session_id)})

@app.route("/api/stats")
def stats():
    return jsonify(brain.get_stats())

@app.route("/api/documents")
def documents():
    docs = []
    for fname, meta in brain.documents_metadata.items():
        docs.append({
            "filename": fname,
            "pages": meta.get("pages", 0),
            "chunks": meta.get("chunks", 0),
            "category": meta.get("category", "general")
        })
    return jsonify({"documents": docs, "total": len(docs)})

@app.route("/api/categories")
def categories():
    return jsonify({"categories": brain.get_categories()})

@app.route("/api/conversation/<session_id>", methods=["DELETE"])
def clear_conversation(session_id):
    memory.clear_session(session_id)
    session_focus.pop(session_id, None)
    return jsonify({"success": True})

@app.route('/widget.js')
def serve_widget():
    """Serve the widget JavaScript file"""
    try:
        widget_path = os.path.join(os.path.dirname(__file__), 'widget.js')
        if os.path.exists(widget_path):
            with open(widget_path, 'r') as f:
                content = f.read()
            return content, 200, {'Content-Type': 'application/javascript'}
        return "Widget not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    print("\n🚀 Pinecone RAG Chatbot: http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)