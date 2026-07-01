# app.py – Pinecone RAG Chatbot (With sentence-transformers)
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("🚀 APP STARTING (With sentence-transformers)")
print("=" * 60)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.debug = True

# ===== CORS CONFIGURATION =====
CORS(app, 
     resources={
         r"/api/*": {
             "origins": "*",
             "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
             "allow_headers": ["Content-Type", "Authorization", "Accept", "X-Requested-With"],
             "expose_headers": ["Content-Type"],
             "supports_credentials": True,
             "max_age": 3600
         }
     })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# ===== CONFIGURATION =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_CHAT_MODEL", "mixtral-8x7b-32768")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in .env file")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env file")

# ===== LOAD EMBEDDING MODEL =====
print("📥 Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

# ===== INITIALIZE CLIENTS =====
print("🔗 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(host=PINECONE_INDEX_HOST)
print(f"✅ Pinecone connected: {PINECONE_INDEX_NAME}")

print("🔗 Connecting to Groq...")
groq_client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq connected")

memory = ConversationMemory()
session_focus = {}

# ===== LOAD METADATA =====
def load_document_metadata():
    try:
        possible_paths = [
            "./brain_metadata.json",
            "brain_metadata.json",
            "/app/brain_metadata.json",
            os.path.join(os.path.dirname(__file__), "brain_metadata.json")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    print(f"✅ Loaded metadata from {path}: {len(data)} documents")
                    return data
        print("⚠️ No metadata file found.")
        return {}
    except Exception as e:
        print(f"⚠️ Error loading metadata: {e}")
        return {}

documents_metadata = load_document_metadata()

def get_all_filenames():
    return list(documents_metadata.keys())

def get_document_count():
    return len(documents_metadata)

# ===== EMBEDDING FUNCTION =====
def get_embedding(text: str):
    """Get embedding using sentence-transformers"""
    if len(text) > 8000:
        text = text[:8000]
    return embedding_model.encode(text).tolist()

# ===== PINECONE SEARCH =====
def search_pinecone(query: str, top_k: int = 5):
    try:
        query_embedding = get_embedding(query)
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.get('matches', [])
    except Exception as e:
        print(f"Search error: {e}")
        raise

# ===== GROQ RESPONSE =====
def generate_response(query: str, context: str):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain the answer, say "I could not find that information in the documents."
Be concise and accurate. Use the context to support your answers.
Always cite the source document when possible."""},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Generation error: {e}")
        raise

# ===== FOCUS MANAGEMENT =====
def resolve_filename(user_input: str):
    all_files = get_all_filenames()
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

# ===== ROUTES =====
@app.route('/widget.js')
def serve_widget():
    """Serve the widget JavaScript file"""
    try:
        return send_from_directory('.', 'widget.js')
    except Exception as e:
        return """// Chat Widget - Knowledge Brain (Full Version)
(function() {
    'use strict';

    const CONFIG = {
        apiUrl: window.CHATBOT_API_URL || window.location.origin,
        botName: window.CHATBOT_NAME || 'Knowledge Bot',
        botAvatar: window.CHATBOT_AVATAR || '🧠',
        primaryColor: window.CHATBOT_COLOR || '#533483',
        greeting: window.CHATBOT_GREETING || 'Hello! Ask me anything about our documents.',
    };

    console.log('🧠 Chat widget loaded');
    console.log('📨 API URL:', CONFIG.apiUrl);

    let sessionId = localStorage.getItem('chatbot_session') || 'session_' + Date.now();
    let isOpen = false;
    let isLoading = false;

    function createWidget() {
        const widget = document.createElement('div');
        widget.id = 'chatbot-widget';
        widget.innerHTML = `
            <style>
                #chatbot-widget * { box-sizing: border-box; margin: 0; padding: 0; }
                .chatbot-button {
                    position: fixed; bottom: 20px; right: 20px;
                    width: 60px; height: 60px; border-radius: 50%;
                    background: ${CONFIG.primaryColor}; color: white;
                    border: none; cursor: pointer; font-size: 24px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    z-index: 9999; transition: transform 0.3s;
                    display: flex; align-items: center; justify-content: center;
                }
                .chatbot-button:hover { transform: scale(1.1); }
                .chatbot-button.hidden { display: none; }
                .chatbot-window {
                    position: fixed; bottom: 90px; right: 20px;
                    width: 380px; height: 500px;
                    background: #16213e; border-radius: 16px;
                    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
                    z-index: 9999; display: none;
                    flex-direction: column; overflow: hidden;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                .chatbot-window.open { display: flex; }
                .chatbot-header {
                    background: linear-gradient(135deg, #0f3460, ${CONFIG.primaryColor});
                    color: white; padding: 16px 20px; font-weight: bold;
                    font-size: 16px; display: flex; align-items: center; gap: 10px;
                }
                .chatbot-header-buttons { margin-left: auto; display: flex; gap: 8px; }
                .chatbot-header-btn {
                    background: rgba(255,255,255,0.2); border: none; color: white;
                    width: 28px; height: 28px; border-radius: 6px;
                    cursor: pointer; font-size: 14px;
                    display: flex; align-items: center; justify-content: center;
                }
                .chatbot-header-btn:hover { background: rgba(255,255,255,0.3); }
                .chatbot-messages {
                    flex: 1; overflow-y: auto; padding: 16px;
                    display: flex; flex-direction: column; gap: 12px;
                }
                .chatbot-message { display: flex; gap: 8px; max-width: 85%; animation: fadeIn 0.3s; }
                @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
                .chatbot-message.user { align-self: flex-end; flex-direction: row-reverse; }
                .chatbot-avatar {
                    width: 30px; height: 30px; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 14px; flex-shrink: 0;
                }
                .chatbot-message.bot .chatbot-avatar { background: ${CONFIG.primaryColor}; }
                .chatbot-message.user .chatbot-avatar { background: #0f3460; }
                .chatbot-message-content {
                    padding: 10px 14px; border-radius: 12px;
                    font-size: 14px; line-height: 1.5; color: white;
                    word-wrap: break-word;
                }
                .chatbot-message.bot .chatbot-message-content { background: #1a1a3e; }
                .chatbot-message.user .chatbot-message-content { background: ${CONFIG.primaryColor}; }
                .chatbot-sources { margin-top: 4px; font-size: 10px; color: #a0aec0; font-style: italic; }
                .chatbot-input-area {
                    display: flex; padding: 12px; border-top: 1px solid #1a1a3e; gap: 8px;
                }
                .chatbot-input {
                    flex: 1; padding: 10px 14px; border: 1px solid #2d2d5e;
                    border-radius: 20px; background: #1a1a3e; color: white;
                    font-size: 14px; outline: none; font-family: inherit;
                }
                .chatbot-input:focus { border-color: ${CONFIG.primaryColor}; }
                .chatbot-send-btn {
                    padding: 10px 18px; background: ${CONFIG.primaryColor};
                    color: white; border: none; border-radius: 20px;
                    cursor: pointer; font-size: 14px; font-family: inherit;
                }
                .chatbot-send-btn:hover { opacity: 0.9; }
                .chatbot-typing { display: flex; gap: 4px; padding: 10px 14px; }
                .chatbot-typing span {
                    width: 8px; height: 8px; border-radius: 50%;
                    background: ${CONFIG.primaryColor};
                    animation: bounce 1.4s infinite;
                }
                .chatbot-typing span:nth-child(2) { animation-delay: 0.2s; }
                .chatbot-typing span:nth-child(3) { animation-delay: 0.4s; }
                @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-6px); } }
                @media (max-width: 480px) {
                    .chatbot-window { width: 100%; height: 100%; bottom: 0; right: 0; border-radius: 0; }
                }
            </style>
            <button class="chatbot-button" id="chatbot-toggle">${CONFIG.botAvatar}</button>
            <div class="chatbot-window" id="chatbot-window">
                <div class="chatbot-header">
                    <span>${CONFIG.botAvatar}</span>
                    ${CONFIG.botName}
                    <div class="chatbot-header-buttons">
                        <button class="chatbot-header-btn" id="chatbot-clear">🔄</button>
                        <button class="chatbot-header-btn" id="chatbot-close">✕</button>
                    </div>
                </div>
                <div class="chatbot-messages" id="chatbot-messages">
                    <div class="chatbot-message bot">
                        <div class="chatbot-avatar">${CONFIG.botAvatar}</div>
                        <div class="chatbot-message-content">${CONFIG.greeting}</div>
                    </div>
                </div>
                <div class="chatbot-input-area">
                    <input type="text" class="chatbot-input" id="chatbot-input" placeholder="Ask a question..." autofocus>
                    <button class="chatbot-send-btn" id="chatbot-send">Send</button>
                </div>
            </div>
        `;
        document.body.appendChild(widget);

        document.getElementById('chatbot-toggle').addEventListener('click', toggleChat);
        document.getElementById('chatbot-close').addEventListener('click', closeChat);
        document.getElementById('chatbot-clear').addEventListener('click', clearChat);
        document.getElementById('chatbot-send').addEventListener('click', sendMessage);
        document.getElementById('chatbot-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

    function toggleChat() {
        isOpen = !isOpen;
        const window = document.getElementById('chatbot-window');
        const button = document.getElementById('chatbot-toggle');
        if (isOpen) {
            window.classList.add('open');
            button.classList.add('hidden');
            document.getElementById('chatbot-input').focus();
        } else {
            window.classList.remove('open');
            button.classList.remove('hidden');
        }
    }

    function closeChat() {
        isOpen = false;
        document.getElementById('chatbot-window').classList.remove('open');
        document.getElementById('chatbot-toggle').classList.remove('hidden');
    }

    async function clearChat() {
        try {
            await fetch(`${CONFIG.apiUrl}/api/conversation/${sessionId}`, { method: 'DELETE' });
            sessionId = 'session_' + Date.now();
            localStorage.setItem('chatbot_session', sessionId);
        } catch(e) {}
        document.getElementById('chatbot-messages').innerHTML = `
            <div class="chatbot-message bot">
                <div class="chatbot-avatar">${CONFIG.botAvatar}</div>
                <div class="chatbot-message-content">Chat cleared. Ask me anything!</div>
            </div>
        `;
    }

    function addMessage(text, role, sources = []) {
        const messagesDiv = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = `chatbot-message ${role}`;
        let html = `<div class="chatbot-avatar">${role === 'user' ? '👤' : CONFIG.botAvatar}</div>`;
        html += `<div class="chatbot-message-content">${text.replace(/\\n/g, '<br>')}`;
        if (sources && sources.length > 0) {
            html += '<div class="chatbot-sources">';
            const seen = new Set();
            sources.forEach(s => {
                if (!seen.has(s.document)) {
                    seen.add(s.document);
                    html += `📄 ${s.document} `;
                }
            });
            html += '</div>';
        }
        html += '</div>';
        div.innerHTML = html;
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function showTyping() {
        const messagesDiv = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = 'chatbot-message bot';
        div.id = 'chatbot-typing';
        div.innerHTML = `
            <div class="chatbot-avatar">${CONFIG.botAvatar}</div>
            <div class="chatbot-message-content">
                <div class="chatbot-typing"><span></span><span></span><span></span></div>
            </div>
        `;
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById('chatbot-typing');
        if (el) el.remove();
    }

    async function sendMessage() {
        const input = document.getElementById('chatbot-input');
        const question = input.value.trim();
        if (!question || isLoading) return;

        console.log('📨 Sending message:', question);
        isLoading = true;
        addMessage(question, 'user');
        input.value = '';
        showTyping();

        try {
            const url = `${CONFIG.apiUrl}/api/chat`;
            const res = await fetch(url, {
                method: 'POST',
                mode: 'cors',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ question, session_id: sessionId })
            });

            const data = await res.json();
            hideTyping();
            if (data.answer) {
                addMessage(data.answer, 'bot', data.sources || []);
            } else {
                addMessage('Sorry, I could not process that question.', 'bot');
            }
        } catch (e) {
            console.error('❌ Widget error:', e);
            hideTyping();
            addMessage('Sorry, an error occurred. Please try again.', 'bot');
        }
        isLoading = false;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createWidget);
    } else {
        createWidget();
    }
})();""", 200, {'Content-Type': 'application/javascript'}

@app.route("/api/ping", methods=["GET", "POST", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response
    return jsonify({"status": "ok", "message": "pong"})

@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/api/chat", methods=["GET", "POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response

    if request.method == "GET":
        return jsonify({"error": "Use POST method"}), 405

    try:
        if not request.is_json:
            return jsonify({"answer": "⚠️ Please send JSON data"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"answer": "⚠️ No data received"}), 400

        question = data.get("question", "").strip()
        session_id = data.get("session_id", session.get("session_id", "default"))

        if not question:
            return jsonify({"answer": "⚠️ Please provide a question."}), 400

        # Search Pinecone
        matches = search_pinecone(question, top_k=5)

        if not matches:
            return jsonify({
                "answer": "I could not find relevant information in the knowledge base.",
                "sources": []
            })

        # Build context
        context_parts = []
        sources = []
        for match in matches:
            if match.get('score', 0) > 0.3:
                text = match.get('metadata', {}).get('text', '')
                source_file = match.get('metadata', {}).get('source_file', 'unknown')
                if text:
                    context_parts.append(text)
                    sources.append({
                        'document': source_file,
                        'score': match.get('score', 0)
                    })

        if not context_parts:
            return jsonify({
                "answer": "I found some information but with low confidence. Please rephrase your question.",
                "sources": []
            })

        context = "\n\n---\n\n".join(context_parts[:3])

        # Generate response
        answer = generate_response(question, context)

        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", answer)

        return jsonify({
            "answer": answer,
            "sources": sources[:3],
            "sources_count": len(sources)
        })

    except Exception as e:
        print(f"❌ Chat error: {e}")
        traceback.print_exc()
        return jsonify({"answer": f"⚠️ Error: {str(e)[:150]}"}), 500

@app.route("/api/stats")
def stats():
    try:
        stats = pinecone_index.describe_index_stats()
        return jsonify({
            "total_documents": get_document_count(),
            "total_chunks": stats.get('total_vector_count', 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents")
def documents():
    docs = []
    for fname, meta in documents_metadata.items():
        docs.append({
            "filename": fname,
            "pages": meta.get("pages", 0),
            "chunks": meta.get("chunks", 0),
            "category": meta.get("category", "general")
        })
    return jsonify({"documents": docs, "total": len(docs)})

@app.route("/api/categories")
def categories():
    cats = set()
    for meta in documents_metadata.values():
        cats.add(meta.get("category", "general"))
    return jsonify({"categories": sorted(list(cats))})

@app.route("/api/conversation/<session_id>", methods=["DELETE"])
def clear_conversation(session_id):
    memory.clear_session(session_id)
    session_focus.pop(session_id, None)
    return jsonify({"success": True})

@app.route("/api/debug", methods=["GET"])
def debug():
    try:
        stats = pinecone_index.describe_index_stats()
        return jsonify({
            "pinecone_connected": True,
            "groq_connected": True,
            "documents_loaded": get_document_count(),
            "vector_count": stats.get('total_vector_count', 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n🚀 Pinecone RAG Chatbot")
    print(f"📚 Documents: {get_document_count()}")
    app.run(debug=False, host="0.0.0.0", port=5000)