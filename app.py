# app.py – Complete with Embedded Widget, Static Serving, and All Routes
from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import re
from flask import Flask, request, jsonify, render_template, session, Response
from brain import KnowledgeBrain
from memory import ConversationMemory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

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

PROMPT = """You are a precise and insightful research assistant.  
Your answers are based **only** on the document context below.

**Rules:**
- If the context contains enough detail, provide a **comprehensive answer** with examples, comparisons, or steps (as appropriate).
- Structure your answer clearly: use **bullet points** for lists, **paragraphs** for explanations.
- Do **not** mention the context itself (e.g., "according to the document").
- If the context lacks the answer, say: "I could not find that information in the knowledge base."
- Match the level of detail to the question: a simple question gets a concise answer; a complex or open‑ended question gets a thorough one.
- **Never** repeat the same sentence more than once.

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

# ============================================================
# WIDGET JAVASCRIPT (Embedded directly in route)
# ============================================================
WIDGET_JS = r'''
// Chat Widget v1.0 - Knowledge Brain Chatbot
(function() {
    const CONFIG = {
        apiUrl: window.CHATBOT_API_URL || 'https://advanced-d5cl1pcg0-gat6.vercel.app',
        botName: window.CHATBOT_NAME || 'Knowledge Bot',
        botAvatar: window.CHATBOT_AVATAR || '🧠',
        primaryColor: window.CHATBOT_COLOR || '#533483',
        greeting: window.CHATBOT_GREETING || 'Hello! Ask me anything about our documents.',
    };

    let sessionId = localStorage.getItem('chatbot_session') || 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('chatbot_session', sessionId);
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
                    background: ${CONFIG.primaryColor};
                    color: white; border: none; cursor: pointer;
                    font-size: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    z-index: 9999; display: flex; align-items: center; justify-content: center;
                    transition: transform 0.3s;
                }
                .chatbot-button:hover { transform: scale(1.1); }
                .chatbot-button.hidden { display: none; }
                .chatbot-window {
                    position: fixed; bottom: 90px; right: 20px;
                    width: 380px; height: 500px;
                    background: #16213e; border-radius: 16px;
                    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
                    z-index: 9999; display: none; flex-direction: column;
                    overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                .chatbot-window.open { display: flex; }
                .chatbot-header {
                    background: linear-gradient(135deg, #0f3460, ${CONFIG.primaryColor});
                    color: white; padding: 16px 20px; font-weight: bold; font-size: 16px;
                    display: flex; align-items: center; gap: 10px;
                }
                .chatbot-header-buttons { margin-left: auto; display: flex; gap: 8px; }
                .chatbot-header-btn {
                    background: rgba(255,255,255,0.2); border: none; color: white;
                    width: 28px; height: 28px; border-radius: 6px; cursor: pointer;
                    font-size: 14px; display: flex; align-items: center; justify-content: center;
                }
                .chatbot-header-btn:hover { background: rgba(255,255,255,0.3); }
                .chatbot-messages {
                    flex: 1; overflow-y: auto; padding: 16px;
                    display: flex; flex-direction: column; gap: 12px;
                }
                .chatbot-message { display: flex; gap: 8px; max-width: 85%; animation: chatbotFadeIn 0.3s; }
                @keyframes chatbotFadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
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
                    font-size: 14px; line-height: 1.5; color: white; word-wrap: break-word;
                }
                .chatbot-message.bot .chatbot-message-content { background: #1a1a3e; }
                .chatbot-message.user .chatbot-message-content { background: ${CONFIG.primaryColor}; }
                .chatbot-sources { margin-top: 4px; font-size: 10px; color: #a0aec0; font-style: italic; }
                .chatbot-input-area { display: flex; padding: 12px; border-top: 1px solid #1a1a3e; gap: 8px; }
                .chatbot-input {
                    flex: 1; padding: 10px 14px; border: 1px solid #2d2d5e;
                    border-radius: 20px; background: #1a1a3e; color: white;
                    font-size: 14px; outline: none; font-family: inherit;
                }
                .chatbot-input:focus { border-color: ${CONFIG.primaryColor}; }
                .chatbot-send-btn {
                    padding: 10px 18px; background: ${CONFIG.primaryColor};
                    color: white; border: none; border-radius: 20px; cursor: pointer;
                    font-size: 14px; font-family: inherit;
                }
                .chatbot-send-btn:hover { opacity: 0.9; }
                .chatbot-typing { display: flex; gap: 4px; padding: 10px 14px; }
                .chatbot-typing span {
                    width: 8px; height: 8px; border-radius: 50%;
                    background: ${CONFIG.primaryColor};
                    animation: chatbotBounce 1.4s infinite;
                }
                .chatbot-typing span:nth-child(2) { animation-delay: 0.2s; }
                .chatbot-typing span:nth-child(3) { animation-delay: 0.4s; }
                @keyframes chatbotBounce {
                    0%, 60%, 100% { transform: translateY(0); }
                    30% { transform: translateY(-6px); }
                }
                @media (max-width: 480px) {
                    .chatbot-window { width: 100%; height: 100%; bottom: 0; right: 0; border-radius: 0; }
                }
            </style>
            <button class="chatbot-button" id="chatbot-toggle">${CONFIG.botAvatar}</button>
            <div class="chatbot-window" id="chatbot-window">
                <div class="chatbot-header">
                    <span>${CONFIG.botAvatar}</span> ${CONFIG.botName}
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
        document.getElementById('chatbot-input').addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
    }

    function toggleChat() {
        isOpen = !isOpen;
        document.getElementById('chatbot-window').classList.toggle('open', isOpen);
        document.getElementById('chatbot-toggle').classList.toggle('hidden', isOpen);
        if (isOpen) document.getElementById('chatbot-input').focus();
    }

    function closeChat() {
        isOpen = false;
        document.getElementById('chatbot-window').classList.remove('open');
        document.getElementById('chatbot-toggle').classList.remove('hidden');
    }

    async function clearChat() {
        try { await fetch(CONFIG.apiUrl + '/api/conversation/' + sessionId, { method: 'DELETE' }); } catch(e) {}
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('chatbot_session', sessionId);
        document.getElementById('chatbot-messages').innerHTML = '<div class="chatbot-message bot"><div class="chatbot-avatar">' + CONFIG.botAvatar + '</div><div class="chatbot-message-content">Chat cleared! Ask me anything.</div></div>';
    }

    function addMessage(text, role, sources) {
        sources = sources || [];
        const div = document.createElement('div');
        div.className = 'chatbot-message ' + role;
        let html = '<div class="chatbot-avatar">' + (role === 'user' ? '👤' : CONFIG.botAvatar) + '</div>';
        html += '<div class="chatbot-message-content">' + text.replace(/\n/g, '<br>');
        if (sources.length > 0) {
            html += '<div class="chatbot-sources">📄 ' + [...new Set(sources.map(s => s.document))].join(', ') + '</div>';
        }
        html += '</div>';
        div.innerHTML = html;
        document.getElementById('chatbot-messages').appendChild(div);
        document.getElementById('chatbot-messages').scrollTop = document.getElementById('chatbot-messages').scrollHeight;
    }

    function showTyping() {
        const div = document.createElement('div');
        div.className = 'chatbot-message bot';
        div.id = 'chatbot-typing';
        div.innerHTML = '<div class="chatbot-avatar">' + CONFIG.botAvatar + '</div><div class="chatbot-message-content"><div class="chatbot-typing"><span></span><span></span><span></span></div></div>';
        document.getElementById('chatbot-messages').appendChild(div);
        document.getElementById('chatbot-messages').scrollTop = document.getElementById('chatbot-messages').scrollHeight;
    }

    function hideTyping() { const el = document.getElementById('chatbot-typing'); if (el) el.remove(); }

    async function sendMessage() {
        const input = document.getElementById('chatbot-input');
        const question = input.value.trim();
        if (!question || isLoading) return;
        isLoading = true;
        addMessage(question, 'user');
        input.value = '';
        showTyping();
        try {
            const res = await fetch(CONFIG.apiUrl + '/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question, session_id: sessionId })
            });
            const data = await res.json();
            hideTyping();
            addMessage(data.answer || 'Sorry, an error occurred.', 'bot', data.sources);
        } catch(e) {
            hideTyping();
            addMessage('Connection error. Please try again.', 'bot');
        }
        isLoading = false;
    }

    if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', createWidget); }
    else { createWidget(); }
})();
'''

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/widget.js")
def serve_widget():
    """Serve the chat widget JavaScript."""
    return Response(WIDGET_JS, mimetype='application/javascript')

@app.route("/widget-demo")
def widget_demo():
    """Serve a demo page for the widget."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Widget Demo</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            pre { background: #1a1a2e; color: #00ff88; padding: 20px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🧠 Chat Widget Demo</h1>
            <p>Click the brain emoji in the bottom-right corner to test the chatbot!</p>
            <h3>Embed Code:</h3>
            <pre><code>&lt;script src="https://advanced-d5cl1pcg0-gat6.vercel.app/widget.js"&gt;&lt;/script&gt;</code></pre>
        </div>
        <script src="/widget.js"></script>
    </body>
    </html>
    '''

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

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Pinecone RAG Chatbot: http://127.0.0.1:5000")
    print("📦 Widget: http://127.0.0.1:5000/widget.js")
    print("📄 Demo: http://127.0.0.1:5000/widget-demo")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=5000)