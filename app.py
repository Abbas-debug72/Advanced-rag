# app.py – RAG Chatbot with User‑specific API Keys
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import uuid
import re
import json
import logging
import traceback
from functools import wraps
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, make_response
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import jwt

from memory import ConversationMemory

# Force flush for logging
sys.stdout.flush()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("🚀 STARTING RAG CHATBOT (with API Keys)")
print("=" * 60)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ===== CORS =====
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key, Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# ===== SUPABASE CONFIGURATION =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_JWT_SECRET:
    raise RuntimeError("Supabase environment variables not set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== HELPERS =====
def get_token_from_request():
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ')[1]
    token = request.cookies.get('chatbot_token')
    if token:
        return token
    return None

def get_api_key_from_request():
    return request.headers.get('X-API-Key')

# ===== AUTH DECORATORS =====
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        # 1. Try JWT / cookie (for dashboard)
        token = get_token_from_request()
        if token:
            try:
                user = supabase.auth.get_user(token)
                if user and user.user:
                    request.user = user.user
                    return f(*args, **kwargs)
            except:
                pass

        # 2. Try API key (for widget)
        api_key = get_api_key_from_request()
        if api_key:
            try:
                # Look up user by api_key in the users table
                result = supabase.table('users').select('*').eq('api_key', api_key).execute()
                if result.data and len(result.data) > 0:
                    user_data = result.data[0]
                    # Get the full user from auth to attach
                    user = supabase.auth.admin.get_user_by_id(user_data['id'])
                    if user and user.user:
                        request.user = user.user
                        return f(*args, **kwargs)
            except Exception as e:
                print(f"API key lookup error: {e}")

        return jsonify({"error": "Missing or invalid authentication"}), 401

    return decorated

# ===== CONFIGURATION =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# ===== LOAD EMBEDDING MODEL =====
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
def get_all_filenames(): return list(documents_metadata.keys())
def get_document_count(): return len(documents_metadata)

# ===== EMBEDDING & SEARCH =====
def get_embedding(text: str):
    if len(text) > 8000: text = text[:8000]
    return embedding_model.encode(text).tolist()

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

def detect_focus_command(question):
    q = question.lower()
    if "clear focus" in q: return "CLEAR"
    match = re.search(r'only\s+use\s+([\w\-.]+(?:\.pdf)?)', q)
    return match.group(1) if match else None

# ===== USER MANAGEMENT =====
def ensure_user_has_api_key(user_id, email):
    """Check if user has an API key; if not, generate one."""
    try:
        result = supabase.table('users').select('api_key').eq('id', user_id).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]['api_key']
        else:
            # Generate new API key
            api_key = str(uuid.uuid4()).replace('-', '')[:32]
            supabase.table('users').insert({
                'id': user_id,
                'email': email,
                'api_key': api_key
            }).execute()
            return api_key
    except Exception as e:
        print(f"Error ensuring API key: {e}")
        return None

# ===== AUTH ROUTES =====

@app.route('/login')
def login_page():
    token = request.cookies.get('chatbot_token')
    if token:
        try:
            user = supabase.auth.get_user(token)
            if user and user.user:
                return redirect('/dashboard')
        except:
            pass
    return render_template("login.html")

@app.route('/signup')
def signup_page():
    token = request.cookies.get('chatbot_token')
    if token:
        try:
            user = supabase.auth.get_user(token)
            if user and user.user:
                return redirect('/dashboard')
        except:
            pass
    return render_template("signup.html")

@app.route('/api/signup', methods=['POST', 'OPTIONS'])
def signup():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })

        if response.user:
            # Create user record with API key
            ensure_user_has_api_key(response.user.id, response.user.email)
            return jsonify({
                "user": {
                    "email": response.user.email,
                    "id": response.user.id
                }
            })
        else:
            return jsonify({"error": "Sign-up failed"}), 400
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if response.user:
            # Ensure API key exists
            ensure_user_has_api_key(response.user.id, response.user.email)
            resp = make_response(jsonify({
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "user": {
                    "email": response.user.email,
                    "id": response.user.id
                }
            }))
            resp.set_cookie('chatbot_token', response.session.access_token,
                            httponly=True, secure=False, samesite='Lax', max_age=60*60*24*7)
            return resp
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": str(e)}), 401

@app.route('/api/logout', methods=['POST'])
@require_auth
def logout():
    try:
        supabase.auth.sign_out()
        resp = jsonify({"success": True})
        resp.set_cookie('chatbot_token', '', expires=0)
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/me', methods=['GET'])
@require_auth
def get_user():
    return jsonify({
        "user": {
            "email": request.user.email,
            "id": request.user.id
        }
    })

@app.route('/api/api_key', methods=['GET'])
@require_auth
def get_api_key():
    user_id = request.user.id
    try:
        result = supabase.table('users').select('api_key').eq('id', user_id).execute()
        if result.data and len(result.data) > 0:
            return jsonify({"api_key": result.data[0]['api_key']})
        else:
            api_key = ensure_user_has_api_key(user_id, request.user.email)
            if api_key:
                return jsonify({"api_key": api_key})
            else:
                return jsonify({"error": "Could not retrieve or create API key"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== DASHBOARD =====
@app.route('/dashboard')
@require_auth
def dashboard():
    return render_template("dashboard.html", user=request.user)

# ===== WIDGET ROUTE (public) =====
@app.route('/widget.js')
def serve_widget():
    widget_code = """
// Chat Widget – with API Key authentication
(function() {
    'use strict';

    const CONFIG = {
        apiUrl: window.CHATBOT_API_URL || window.location.origin,
        botName: window.CHATBOT_NAME || 'Knowledge Bot',
        botAvatar: window.CHATBOT_AVATAR || '🧠',
        primaryColor: window.CHATBOT_COLOR || '#6C63FF',
        secondaryColor: '#3F3D56',
        greeting: window.CHATBOT_GREETING || 'Hello! Ask me anything about our documents.',
        apiKey: window.CHATBOT_API_KEY || null,   // Required
    };

    console.log('🧠 Chat widget loaded');
    let sessionId = localStorage.getItem('chatbot_session') || 'session_' + Date.now();
    let isOpen = false;
    let isLoading = false;

    // ── Check API key ──
    function hasApiKey() {
        return CONFIG.apiKey && CONFIG.apiKey.length > 0;
    }

    // ── Create widget ──
    function createWidget() {
        const widget = document.createElement('div');
        widget.id = 'chatbot-widget';
        widget.innerHTML = `
            <style>
                /* ... same as before ... */
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

        // Check API key
        if (!hasApiKey()) {
            const msgs = document.getElementById('chatbot-messages');
            msgs.innerHTML = `
                <div class="chatbot-message bot">
                    <div class="avatar">${CONFIG.botAvatar}</div>
                    <div class="bubble">❌ Missing API key. Please set window.CHATBOT_API_KEY.</div>
                </div>
            `;
            document.getElementById('chatbot-input').disabled = true;
            document.getElementById('chatbot-send').disabled = true;
        } else {
            document.getElementById('chatbot-input').disabled = false;
            document.getElementById('chatbot-send').disabled = false;
        }

        document.getElementById('chatbot-toggle').addEventListener('click', toggleChat);
        document.getElementById('chatbot-close').addEventListener('click', closeChat);
        document.getElementById('chatbot-clear').addEventListener('click', clearChat);
        document.getElementById('chatbot-send').addEventListener('click', sendMessage);
        document.getElementById('chatbot-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

    function toggleChat() {
        if (!hasApiKey()) {
            alert('Please set window.CHATBOT_API_KEY to use the widget.');
            return;
        }
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
            await fetch(`${CONFIG.apiUrl}/api/conversation/${sessionId}`, {
                method: 'DELETE',
                headers: { 'X-API-Key': CONFIG.apiKey }
            });
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
        if (!hasApiKey()) {
            alert('Missing API key.');
            return;
        }

        isLoading = true;
        addMessage(question, 'user');
        input.value = '';
        showTyping();

        try {
            const res = await fetch(`${CONFIG.apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': CONFIG.apiKey
                },
                body: JSON.stringify({ question, session_id: sessionId })
            });

            const data = await res.json();

            if (res.status === 401) {
                hideTyping();
                addMessage('🔐 Invalid API key. Please check your configuration.', 'bot');
                document.getElementById('chatbot-input').disabled = true;
                document.getElementById('chatbot-send').disabled = true;
                isLoading = false;
                return;
            }

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

# ===== MAIN PAGE =====
@app.route('/')
def index():
    token = request.cookies.get('chatbot_token')
    if token:
        try:
            user = supabase.auth.get_user(token)
            if user and user.user:
                return redirect('/dashboard')
        except:
            pass
    return redirect('/login')

# ===== PROTECTED CHAT API =====
@app.route("/api/chat", methods=["POST", "OPTIONS"])
@require_auth
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

        # Search Pinecone
        matches = search_pinecone(question, top_k=15)
        if not matches:
            return jsonify({"answer": "I could not find any matching chunks.", "sources": []})

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

        context = "\n\n---\n\n".join(context_parts[:5])
        answer = generate_response(question, context)

        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", answer)

        return jsonify({"answer": answer, "sources": sources[:5]})

    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({"answer": f"⚠️ Server error: {str(e)[:100]}"}), 500

# ===== OTHER PROTECTED ENDPOINTS (use @require_auth) =====
@app.route("/api/stats")
@require_auth
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
@require_auth
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
@require_auth
def categories():
    cats = set()
    for meta in documents_metadata.values():
        cats.add(meta.get("category", "general"))
    return jsonify({"categories": sorted(list(cats))})

@app.route("/api/conversation/<session_id>", methods=["DELETE"])
@require_auth
def clear_conversation(session_id):
    memory.clear_session(session_id)
    session_focus.pop(session_id, None)
    return jsonify({"success": True})

@app.route("/api/debug")
@require_auth
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