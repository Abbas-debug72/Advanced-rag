# app.py – Pinecone RAG Chatbot (with improved prompt & full CORS support)
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
from brain import KnowledgeBrain
from memory import ConversationMemory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Force flush for logging
sys.stdout.flush()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("🚀 APP STARTING with DEBUG LOGGING")
print("=" * 60)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.debug = True

# ===== CORS CONFIGURATION =====
# Enable CORS for all routes with comprehensive settings
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

# Manual CORS headers as fallback
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# ===== INITIALIZE KNOWLEDGE BRAIN =====
print("🧠 Loading Knowledge Brain...")
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

# ===== SIMPLE PING ENDPOINT =====
@app.route("/api/ping", methods=["GET", "POST", "OPTIONS"])
def ping():
    """Simple ping endpoint to test if app is responding"""
    print("📨 PING received - method:", request.method)
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response
    return jsonify({
        "status": "ok",
        "message": "pong",
        "method": request.method
    })

# ===== WIDGET ROUTE =====
@app.route('/widget.js')
def serve_widget():
    """Serve the widget JavaScript file"""
    print("📨 Serving widget.js")
    try:
        return send_from_directory('.', 'widget.js')
    except Exception as e:
        print(f"⚠️ Error serving widget.js: {e}")
        # Fallback widget code with full functionality
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
        } catch(e) {
            console.warn('Clear chat error:', e);
        }
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
            console.log('📨 Full URL:', url);

            const res = await fetch(url, {
                method: 'POST',
                mode: 'cors',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ question, session_id: sessionId })
            });

            console.log('📨 Response status:', res.status);
            console.log('📨 Response headers:', res.headers);

            if (!res.ok) {
                const errorText = await res.text();
                console.error('❌ Response error:', errorText);
                throw new Error(`HTTP ${res.status}: ${errorText}`);
            }

            const data = await res.json();
            console.log('📨 Response data:', data);

            hideTyping();
            if (data.answer) {
                addMessage(data.answer, 'bot', data.sources || []);
            } else {
                addMessage('Sorry, I could not process that question.', 'bot');
            }
        } catch (e) {
            console.error('❌ Widget error:', e);
            hideTyping();
            
            let errorMsg = 'Sorry, an error occurred. Please try again.';
            if (e.message === 'Failed to fetch') {
                errorMsg = '⚠️ Cannot connect to the server. Please check your internet connection.';
            } else if (e.message.includes('CORS')) {
                errorMsg = '⚠️ CORS error. Please contact the site administrator.';
            }
            addMessage(errorMsg, 'bot');
        }
        isLoading = false;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createWidget);
    } else {
        createWidget();
    }
})();""", 200, {'Content-Type': 'application/javascript'}

# ===== IMPROVED PROMPT =====
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

# ===== Focus management =====
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

# ===== ROUTES =====
@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/api/chat", methods=["GET", "POST", "OPTIONS"])
def chat():
    """Main chat endpoint with extensive debug logging"""
    
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        print("📨 OPTIONS request received")
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Max-Age', '3600')
        return response

    if request.method == "GET":
        return jsonify({"error": "Use POST method for chat"}), 405

    print(f"\n{'='*60}")
    print(f"📨 CHAT REQUEST STARTED at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    start_time = time.time()

    try:
        # Log request info
        print(f"📨 Request method: {request.method}")
        print(f"📨 Content-Type: {request.content_type}")
        
        # Get raw data
        raw_data = request.get_data(as_text=True)
        print(f"📨 Raw request data: {raw_data[:500]}")
        
        # Try to parse JSON
        data = None
        if request.is_json:
            data = request.get_json()
            print(f"📨 Parsed JSON data: {data}")
        else:
            print(f"⚠️ Request is not JSON. Trying manual parse...")
            try:
                data = json.loads(raw_data) if raw_data else None
                print(f"📨 Manually parsed JSON: {data}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                return jsonify({
                    "error": "Invalid JSON",
                    "answer": "⚠️ Invalid JSON format. Please send valid JSON."
                }), 400
        
        if not data:
            print("❌ No data received")
            return jsonify({
                "error": "No data",
                "answer": "⚠️ No data received. Please send a question."
            }), 400

        question = data.get("question", "").strip()
        session_id = data.get("session_id", session.get("session_id", "default"))
        category = data.get("category", "all")
        
        print(f"📨 Question: '{question}'")
        print(f"📨 Session ID: {session_id}")
        print(f"📨 Category: {category}")

        if not question:
            print("❌ Empty question")
            return jsonify({
                "error": "Question required",
                "answer": "⚠️ Please provide a question."
            }), 400

        # --- Focus detection ---
        focus_file = session_focus.get(session_id)
        focus_cmd = detect_focus_command(question)
        print(f"📨 Current focus: {focus_file}")
        print(f"📨 Focus command: {focus_cmd}")

        if focus_cmd == "CLEAR":
            print("📨 Clearing focus")
            session_focus.pop(session_id, None)
            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "assistant", "✅ Document filter cleared.")
            return jsonify({"answer": "✅ Document filter cleared.", "sources": [], "focus": None})

        if focus_cmd is not None:
            print(f"📨 Setting focus to: {focus_cmd}")
            all_files = brain.get_all_filenames()
            if focus_cmd in all_files:
                session_focus[session_id] = focus_cmd
                msg = f"✅ Now focusing on **{focus_cmd}** only."
            else:
                msg = f"❌ Document `{focus_cmd}` not found. Available: {', '.join(all_files)}"
            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "assistant", msg)
            return jsonify({"answer": msg, "sources": [], "focus": session_focus.get(session_id)})

        # --- retrieval ---
        print("📨 Retrieving conversation history...")
        history = memory.get_history(session_id, last_n=6)
        history_str = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ) if history else "No history"

        print("📨 Searching for relevant documents...")
        try:
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
            print(f"📨 Found {len(docs)} documents")
        except Exception as e:
            print(f"❌ Search error: {e}")
            traceback.print_exc()
            return jsonify({"answer": f"⚠️ Search error: {str(e)[:100]}"}), 500

        context = format_docs(docs)
        focus_info = f"Currently focused on: {focus_file}. Only use this document." if focus_file else "No active document filter."

        # --- generation ---
        print("📨 Generating response...")
        try:
            chain = QA_PROMPT | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "chat_history": history_str,
                "question": question,
                "focus_info": focus_info
            })
            print(f"📨 Generated answer length: {len(answer)}")
        except Exception as e:
            print(f"❌ Generation error: {e}")
            traceback.print_exc()
            return jsonify({"answer": f"⚠️ Generation error: {str(e)[:100]}"}), 500

        # --- Save to memory ---
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", answer)

        # --- Format sources ---
        seen_src = set()
        sources = []
        for doc in docs:
            src = doc.metadata.get("source_file", "?")
            if src not in seen_src:
                seen_src.add(src)
                sources.append({"document": src, "page": doc.metadata.get("page_number", "?")})
        
        elapsed = time.time() - start_time
        print(f"📨 Request completed in {elapsed:.2f} seconds")
        print(f"{'='*60}\n")

        return jsonify({"answer": answer, "sources": sources, "focus": focus_file})

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ CHAT ERROR after {elapsed:.2f} seconds:")
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return jsonify({
            "error": str(e),
            "answer": f"⚠️ An error occurred: {str(e)[:150]}"
        }), 500

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

@app.route("/api/debug", methods=["GET"])
def debug():
    """Debug endpoint to check environment and connections"""
    import os
    return jsonify({
        "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "documents_loaded": len(brain.documents_metadata) if brain else 0,
        "chunks": brain.get_stats().get("total_chunks", 0) if brain else 0,
        "env_vars": [k for k in os.environ.keys() if not k.startswith('_') and not k.startswith('PYTHON')][:10]
    })

if __name__ == "__main__":
    print("\n🚀 Pinecone RAG Chatbot: http://127.0.0.1:5000")
    print("📊 Debug endpoint: /api/debug")
    print("🧪 Test endpoint: /api/ping")
    app.run(debug=False, host="0.0.0.0", port=5000)