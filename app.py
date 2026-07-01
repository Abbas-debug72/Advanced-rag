# app.py – RAG Chatbot (Polished Widget + Active Groq Model)
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("🚀 STARTING RAG CHATBOT (Polished Widget)")
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
GROQ_MODEL = "llama-3.1-8b-instant"   # Confirmed active model

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# ===== LOAD EMBEDDING MODEL (all-MiniLM-L6-v2) =====
print("📥 Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded (384-dim)")

# ===== INITIALIZE CLIENTS =====
print("🔗 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(host=PINECONE_INDEX_HOST)
print(f"✅ Pinecone index: {PINECONE_INDEX_NAME}")

print("🔗 Connecting to Groq...")
groq_client = Groq(api_key=GROQ_API_KEY)
print(f"✅ Groq ready (model: {GROQ_MODEL})")

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

# ===== PINECONE SEARCH (no threshold, get all results) =====
def search_pinecone(query: str, top_k: int = 15):
    try:
        q_emb = get_embedding(query)
        results = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        matches = results.get('matches', [])
        if matches:
            scores = [round(m['score'], 4) for m in matches[:5]]
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
    widget_code = """
// Chat Widget – Polished UI
(function() {
    'use strict';

    const CONFIG = {
        apiUrl: window.CHATBOT_API_URL || window.location.origin,
        botName: window.CHATBOT_NAME || 'Knowledge Bot',
        botAvatar: window.CHATBOT_AVATAR || '🧠',
        primaryColor: window.CHATBOT_COLOR || '#6C63FF',
        secondaryColor: '#3F3D56',
        greeting: window.CHATBOT_GREETING || 'Hello! Ask me anything about our documents.',
    };

    console.log('🧠 Chat widget loaded');
    let sessionId = localStorage.getItem('chatbot_session') || 'session_' + Date.now();
    let isOpen = false;
    let isLoading = false;

    function createWidget() {
        const widget = document.createElement('div');
        widget.id = 'chatbot-widget';
        widget.innerHTML = `
            <style>
                #chatbot-widget * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                #chatbot-widget .chatbot-button {
                    position: fixed;
                    bottom: 24px;
                    right: 24px;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: ${CONFIG.primaryColor};
                    color: #fff;
                    border: none;
                    box-shadow: 0 6px 24px rgba(108, 99, 255, 0.4);
                    cursor: pointer;
                    font-size: 28px;
                    z-index: 99999;
                    transition: transform 0.2s, box-shadow 0.2s;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                #chatbot-widget .chatbot-button:hover {
                    transform: scale(1.08);
                    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.5);
                }
                #chatbot-widget .chatbot-button.hidden {
                    display: none;
                }
                #chatbot-widget .chatbot-window {
                    position: fixed;
                    bottom: 100px;
                    right: 24px;
                    width: 400px;
                    max-width: calc(100vw - 48px);
                    height: 560px;
                    max-height: calc(100vh - 140px);
                    background: #ffffff;
                    border-radius: 20px;
                    box-shadow: 0 16px 60px rgba(0, 0, 0, 0.25);
                    z-index: 99998;
                    display: none;
                    flex-direction: column;
                    overflow: hidden;
                    animation: slideUp 0.3s ease-out;
                }
                @keyframes slideUp {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                #chatbot-widget .chatbot-window.open {
                    display: flex;
                }
                #chatbot-widget .chatbot-header {
                    background: ${CONFIG.primaryColor};
                    color: #fff;
                    padding: 18px 20px;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    flex-shrink: 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                #chatbot-widget .chatbot-header .bot-icon {
                    font-size: 24px;
                }
                #chatbot-widget .chatbot-header .bot-name {
                    font-size: 16px;
                    font-weight: 600;
                    flex: 1;
                }
                #chatbot-widget .chatbot-header .header-actions {
                    display: flex;
                    gap: 6px;
                }
                #chatbot-widget .chatbot-header .header-btn {
                    background: rgba(255,255,255,0.15);
                    border: none;
                    color: #fff;
                    width: 32px;
                    height: 32px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 16px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: background 0.2s;
                }
                #chatbot-widget .chatbot-header .header-btn:hover {
                    background: rgba(255,255,255,0.3);
                }
                #chatbot-widget .chatbot-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 16px 16px 8px 16px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                    background: #f8f9fc;
                }
                #chatbot-widget .chatbot-messages::-webkit-scrollbar {
                    width: 4px;
                }
                #chatbot-widget .chatbot-messages::-webkit-scrollbar-track {
                    background: transparent;
                }
                #chatbot-widget .chatbot-messages::-webkit-scrollbar-thumb {
                    background: #d0d5e0;
                    border-radius: 4px;
                }
                #chatbot-widget .chatbot-message {
                    display: flex;
                    gap: 10px;
                    max-width: 85%;
                    animation: fadeIn 0.25s ease;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(8px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                #chatbot-widget .chatbot-message.user {
                    align-self: flex-end;
                    flex-direction: row-reverse;
                }
                #chatbot-widget .chatbot-message .avatar {
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    flex-shrink: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    background: ${CONFIG.primaryColor};
                    color: #fff;
                }
                #chatbot-widget .chatbot-message.user .avatar {
                    background: ${CONFIG.secondaryColor};
                }
                #chatbot-widget .chatbot-message .bubble {
                    padding: 12px 16px;
                    border-radius: 16px;
                    font-size: 14px;
                    line-height: 1.6;
                    word-break: break-word;
                    background: #fff;
                    color: #1e1e2f;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                }
                #chatbot-widget .chatbot-message.bot .bubble {
                    background: #ffffff;
                    border-bottom-left-radius: 4px;
                }
                #chatbot-widget .chatbot-message.user .bubble {
                    background: ${CONFIG.primaryColor};
                    color: #fff;
                    border-bottom-right-radius: 4px;
                }
                #chatbot-widget .chatbot-message .sources {
                    margin-top: 6px;
                    font-size: 11px;
                    color: #8e95a9;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 4px 8px;
                }
                #chatbot-widget .chatbot-message .sources span {
                    background: #f0f2f5;
                    padding: 2px 8px;
                    border-radius: 12px;
                }
                #chatbot-widget .chatbot-typing {
                    display: flex;
                    gap: 4px;
                    padding: 8px 0;
                }
                #chatbot-widget .chatbot-typing span {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: ${CONFIG.primaryColor};
                    animation: bounce 1.4s infinite;
                }
                #chatbot-widget .chatbot-typing span:nth-child(2) { animation-delay: 0.2s; }
                #chatbot-widget .chatbot-typing span:nth-child(3) { animation-delay: 0.4s; }
                @keyframes bounce {
                    0%, 60%, 100% { transform: translateY(0); }
                    30% { transform: translateY(-8px); }
                }
                #chatbot-widget .chatbot-input-area {
                    display: flex;
                    gap: 10px;
                    padding: 12px 16px;
                    background: #fff;
                    border-top: 1px solid #eef0f4;
                    flex-shrink: 0;
                }
                #chatbot-widget .chatbot-input-area input {
                    flex: 1;
                    padding: 10px 14px;
                    border: 1px solid #e2e6ed;
                    border-radius: 24px;
                    font-size: 14px;
                    outline: none;
                    transition: border 0.2s;
                    background: #f8f9fc;
                }
                #chatbot-widget .chatbot-input-area input:focus {
                    border-color: ${CONFIG.primaryColor};
                    background: #fff;
                }
                #chatbot-widget .chatbot-input-area button {
                    padding: 10px 20px;
                    background: ${CONFIG.primaryColor};
                    color: #fff;
                    border: none;
                    border-radius: 24px;
                    font-size: 14px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: background 0.2s;
                    white-space: nowrap;
                }
                #chatbot-widget .chatbot-input-area button:hover {
                    background: #5a52d5;
                }
                @media (max-width: 500px) {
                    #chatbot-widget .chatbot-window {
                        bottom: 0;
                        right: 0;
                        width: 100%;
                        height: 100%;
                        max-height: 100vh;
                        border-radius: 0;
                    }
                    #chatbot-widget .chatbot-button {
                        bottom: 16px;
                        right: 16px;
                        width: 56px;
                        height: 56px;
                        font-size: 24px;
                    }
                }
            </style>

            <button class="chatbot-button" id="chatbot-toggle">${CONFIG.botAvatar}</button>

            <div class="chatbot-window" id="chatbot-window">
                <div class="chatbot-header">
                    <span class="bot-icon">${CONFIG.botAvatar}</span>
                    <span class="bot-name">${CONFIG.botName}</span>
                    <div class="header-actions">
                        <button class="header-btn" id="chatbot-clear" title="Clear chat">↻</button>
                        <button class="header-btn" id="chatbot-close" title="Close">✕</button>
                    </div>
                </div>

                <div class="chatbot-messages" id="chatbot-messages">
                    <div class="chatbot-message bot">
                        <div class="avatar">${CONFIG.botAvatar}</div>
                        <div class="bubble">${CONFIG.greeting}</div>
                    </div>
                </div>

                <div class="chatbot-input-area">
                    <input id="chatbot-input" placeholder="Ask a question..." autofocus>
                    <button id="chatbot-send">Send</button>
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
        const win = document.getElementById('chatbot-window');
        const btn = document.getElementById('chatbot-toggle');
        if (isOpen) {
            win.classList.add('open');
            btn.classList.add('hidden');
            setTimeout(() => document.getElementById('chatbot-input').focus(), 200);
        } else {
            win.classList.remove('open');
            btn.classList.remove('hidden');
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
        } catch(e) {}
        sessionId = 'session_' + Date.now();
        localStorage.setItem('chatbot_session', sessionId);
        document.getElementById('chatbot-messages').innerHTML = `
            <div class="chatbot-message bot">
                <div class="avatar">${CONFIG.botAvatar}</div>
                <div class="bubble">Chat cleared. Ask me anything!</div>
            </div>
        `;
    }

    function addMessage(text, role, sources = []) {
        const container = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = `chatbot-message ${role}`;
        const avatar = role === 'user' ? '👤' : CONFIG.botAvatar;
        let html = `<div class="avatar">${avatar}</div>`;
        html += `<div class="bubble">${text.replace(/\\n/g, '<br>')}`;
        if (sources && sources.length > 0) {
            html += `<div class="sources">`;
            const seen = new Set();
            sources.forEach(s => {
                if (!seen.has(s.document)) {
                    seen.add(s.document);
                    html += `<span>📄 ${s.document}</span>`;
                }
            });
            html += `</div>`;
        }
        html += `</div>`;
        div.innerHTML = html;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    function showTyping() {
        const container = document.getElementById('chatbot-messages');
        const div = document.createElement('div');
        div.className = 'chatbot-message bot';
        div.id = 'chatbot-typing';
        div.innerHTML = `
            <div class="avatar">${CONFIG.botAvatar}</div>
            <div class="bubble"><div class="chatbot-typing"><span></span><span></span><span></span></div></div>
        `;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById('chatbot-typing');
        if (el) el.remove();
    }

    async function sendMessage() {
        const input = document.getElementById('chatbot-input');
        const question = input.value.trim();
        if (!question || isLoading) return;

        isLoading = true;
        addMessage(question, 'user');
        input.value = '';
        showTyping();

        try {
            const res = await fetch(`${CONFIG.apiUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: sessionId })
            });
            const data = await res.json();
            hideTyping();
            if (data.answer) {
                addMessage(data.answer, 'bot', data.sources || []);
            } else {
                addMessage('⚠️ Could not process your question.', 'bot');
            }
        } catch (e) {
            console.error('Widget error:', e);
            hideTyping();
            addMessage('⚠️ Network error. Please try again.', 'bot');
        }
        isLoading = false;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createWidget);
    } else {
        createWidget();
    }
})();
"""
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

        # Search Pinecone – top 15, no score threshold
        matches = search_pinecone(question, top_k=15)
        if not matches:
            return jsonify({"answer": "I could not find any matching chunks.", "sources": []})

        # Build context from all matches
        context_parts = []
        sources = []
        for match in matches:
            text = match.get('metadata', {}).get('text', '')
            src = match.get('metadata', {}).get('source_file', 'unknown')
            if text:
                context_parts.append(text)
                sources.append({"document": src, "score": round(match.get('score', 0), 4)})

        if not context_parts:
            return jsonify({"answer": "Found matches but no text. Please rephrase.", "sources": []})

        # Use top 5 chunks for context
        context = "\n\n---\n\n".join(context_parts[:5])
        answer = generate_response(question, context)

        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", answer)

        return jsonify({"answer": answer, "sources": sources[:5]})

    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({"answer": f"⚠️ Server error: {str(e)[:100]}"}), 500

# ===== OTHER ENDPOINTS =====
@app.route("/api/stats")
def stats():
    try:
        s = pinecone_index.describe_index_stats()
        return jsonify({
            "total_documents": get_document_count(),
            "total_chunks": s.get('total_vector_count', 0)
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

@app.route("/api/debug")
def debug():
    try:
        s = pinecone_index.describe_index_stats()
        return jsonify({
            "pinecone": True,
            "groq": True,
            "model": GROQ_MODEL,
            "documents": get_document_count(),
            "vectors": s.get('total_vector_count', 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n🚀 Server running on http://0.0.0.0:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)