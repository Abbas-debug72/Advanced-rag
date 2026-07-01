# app.py – with relaxed score threshold and debug logs
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import uuid
import re
import time
import json
import logging
import traceback
from flask import Flask, request, jsonify, render_template, session, send_from_directory
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer
from memory import ConversationMemory

# Force flush for logging
sys.stdout.flush()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("🚀 STARTING RAG CHATBOT (Fixed threshold)")
print("=" * 60)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ===== CORS =====
CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# ===== CONFIGURATION =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"   # valid model

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# ===== LOAD EMBEDDING MODEL =====
# Note: This model differs from the ingestion model (BAAI/bge-small-en-v1.5).
# Scores will be lower, so we use a much lower threshold.
print("📥 Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

# ===== INITIALIZE CLIENTS =====
print("🔗 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(host=PINECONE_INDEX_HOST)
print(f"✅ Pinecone index: {PINECONE_INDEX_NAME}")

print("🔗 Connecting to Groq...")
groq_client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq client ready")

memory = ConversationMemory()
session_focus = {}

# ===== LOAD METADATA =====
def load_document_metadata():
    try:
        with open("brain_metadata.json", "r") as f:
            data = json.load(f)
            print(f"✅ Loaded {len(data)} documents from metadata")
            return data
    except Exception as e:
        print(f"⚠️ Could not load metadata: {e}")
        return {}

documents_metadata = load_document_metadata()
def get_all_filenames():
    return list(documents_metadata.keys())
def get_document_count():
    return len(documents_metadata)

# ===== EMBEDDING FUNCTION =====
def get_embedding(text: str):
    if len(text) > 8000:
        text = text[:8000]
    return embedding_model.encode(text).tolist()

# ===== PINECONE SEARCH (with relaxed threshold) =====
def search_pinecone(query: str, top_k: int = 10):
    try:
        q_emb = get_embedding(query)
        results = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        matches = results.get('matches', [])
        # Log scores for debugging
        if matches:
            scores = [m['score'] for m in matches[:5]]
            print(f"📊 Top scores: {scores}")
        return matches
    except Exception as e:
        print(f"Search error: {e}")
        raise

# ===== GROQ CHAT =====
def generate_response(query: str, context: str):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context. If you don't know, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Generation error: {e}")
        raise

# ===== FOCUS COMMANDS =====
def detect_focus_command(question):
    q = question.lower()
    if "clear focus" in q:
        return "CLEAR"
    match = re.search(r'only\s+use\s+([\w\-.]+(?:\.pdf)?)', q)
    if match:
        return match.group(1)
    return None

# ===== ROUTES =====

@app.route('/widget.js')
def serve_widget():
    # (same widget code as before – omitted for brevity, but include it)
    widget_code = """// ... (paste the full widget code from previous answer) ..."""
    return widget_code, 200, {'Content-Type': 'application/javascript'}

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"answer": "⚠️ Please provide a question."}), 400

        question = data['question'].strip()
        session_id = data.get('session_id', session.get('session_id', 'default'))
        if not question:
            return jsonify({"answer": "⚠️ Empty question."}), 400

        # Focus commands
        focus_cmd = detect_focus_command(question)
        if focus_cmd == "CLEAR":
            session_focus.pop(session_id, None)
            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "assistant", "✅ Document filter cleared.")
            return jsonify({"answer": "✅ Document filter cleared.", "sources": []})

        if focus_cmd:
            all_files = get_all_filenames()
            if focus_cmd in all_files:
                session_focus[session_id] = focus_cmd
                msg = f"✅ Now focusing on {focus_cmd}."
            else:
                msg = f"❌ Document '{focus_cmd}' not found."
            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "assistant", msg)
            return jsonify({"answer": msg, "sources": []})

        # Search Pinecone with top_k=10 and very low threshold
        matches = search_pinecone(question, top_k=10)
        if not matches:
            return jsonify({"answer": "I could not find relevant information.", "sources": []})

        # Build context – accept any match (score > 0.0) but log scores
        context_parts = []
        sources = []
        for match in matches:
            score = match.get('score', 0)
            # Lowered threshold to 0.05 to accept almost all matches
            if score > 0.05:
                text = match.get('metadata', {}).get('text', '')
                src = match.get('metadata', {}).get('source_file', 'unknown')
                if text:
                    context_parts.append(text)
                    sources.append({"document": src, "score": score})

        # If still no context, take top 2 regardless of score
        if not context_parts and matches:
            for match in matches[:2]:
                text = match.get('metadata', {}).get('text', '')
                src = match.get('metadata', {}).get('source_file', 'unknown')
                if text:
                    context_parts.append(text)
                    sources.append({"document": src, "score": match.get('score', 0)})

        if not context_parts:
            return jsonify({"answer": "Found results but no text content. Please rephrase.", "sources": []})

        context = "\n\n---\n\n".join(context_parts[:3])
        answer = generate_response(question, context)

        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", answer)

        return jsonify({"answer": answer, "sources": sources[:3]})

    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({"answer": f"⚠️ Server error: {str(e)[:100]}"}), 500

# ... (other endpoints: /api/stats, /api/documents, /api/categories, /api/conversation, /api/debug) remain the same as before ...

if __name__ == "__main__":
    print("\n🚀 Server running on http://0.0.0.0:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)