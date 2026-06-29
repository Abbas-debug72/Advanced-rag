# app.py – Vercel Serverless Optimized
from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import re
import json
from flask import Flask, request, jsonify, Response

# Import brain and memory
from brain import KnowledgeBrain
from memory import ConversationMemory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize brain and LLM once (cached between requests)
print("🧠 Loading Knowledge Brain...")
brain = KnowledgeBrain(pdf_directory=os.getenv("PDF_DIRECTORY", "./pdfs"))

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024)

memory = ConversationMemory()
session_focus = {}

PROMPT = """You are a helpful assistant. Answer based ONLY on the context below.
If the context lacks the answer, say: "I could not find that information in the knowledge base."

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = ChatPromptTemplate.from_template(PROMPT)

def format_docs(docs):
    parts = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get('source_file', '?')
        if src in seen:
            continue
        seen.add(src)
        parts.append(f"[{src}]\n{doc.page_content[:500]}\n")
    return "\n".join(parts)

# ============================================================
# WIDGET JAVASCRIPT
# ============================================================
WIDGET_JS = '''
(function() {
    const apiUrl = window.CHATBOT_API_URL || window.location.origin;
    const botName = window.CHATBOT_NAME || 'Knowledge Bot';
    const botAvatar = window.CHATBOT_AVATAR || '🧠';
    const primaryColor = window.CHATBOT_COLOR || '#533483';
    const greeting = window.CHATBOT_GREETING || 'Hello! Ask me anything.';

    let sessionId = localStorage.getItem('chatbot_session') || 'session_' + Date.now();
    localStorage.setItem('chatbot_session', sessionId);
    let isOpen = false, isLoading = false;

    function createWidget() {
        const widget = document.createElement('div');
        widget.innerHTML = '<style>' +
            '.cw-btn{position:fixed;bottom:20px;right:20px;width:60px;height:60px;border-radius:50%;background:' + primaryColor + ';color:white;border:none;cursor:pointer;font-size:24px;z-index:9999;box-shadow:0 4px 20px rgba(0,0,0,0.3)}' +
            '.cw-btn.hidden{display:none}' +
            '.cw-win{position:fixed;bottom:90px;right:20px;width:380px;height:500px;background:#16213e;border-radius:16px;z-index:9999;display:none;flex-direction:column;overflow:hidden;font-family:sans-serif;box-shadow:0 8px 40px rgba(0,0,0,0.4)}' +
            '.cw-win.open{display:flex}' +
            '.cw-hd{background:linear-gradient(135deg,#0f3460,' + primaryColor + ');color:white;padding:16px;font-weight:bold;display:flex;gap:10px;align-items:center}' +
            '.cw-msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}' +
            '.cw-msg{display:flex;gap:8px;max-width:85%}' +
            '.cw-msg.user{align-self:flex-end;flex-direction:row-reverse}' +
            '.cw-av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}' +
            '.cw-msg.bot .cw-av{background:' + primaryColor + '}' +
            '.cw-msg.user .cw-av{background:#0f3460}' +
            '.cw-txt{padding:10px 14px;border-radius:12px;font-size:14px;color:white;line-height:1.5}' +
            '.cw-msg.bot .cw-txt{background:#1a1a3e}' +
            '.cw-msg.user .cw-txt{background:' + primaryColor + '}' +
            '.cw-inp-area{display:flex;padding:12px;border-top:1px solid #1a1a3e;gap:8px}' +
            '.cw-inp{flex:1;padding:10px;border:1px solid #2d2d5e;border-radius:20px;background:#1a1a3e;color:white;font-size:14px;outline:none}' +
            '.cw-send{padding:10px 18px;background:' + primaryColor + ';color:white;border:none;border-radius:20px;cursor:pointer}' +
            '@media(max-width:480px){.cw-win{width:100%;height:100%;bottom:0;right:0;border-radius:0}}' +
            '</style>' +
            '<button class="cw-btn" id="cw-btn">' + botAvatar + '</button>' +
            '<div class="cw-win" id="cw-win">' +
            '<div class="cw-hd"><span>' + botAvatar + '</span>' + botName + '<span style="margin-left:auto;cursor:pointer" onclick="document.getElementById(\'cw-win\').classList.remove(\'open\');document.getElementById(\'cw-btn\').classList.remove(\'hidden\')">✕</span></div>' +
            '<div class="cw-msgs" id="cw-msgs"><div class="cw-msg bot"><div class="cw-av">' + botAvatar + '</div><div class="cw-txt">' + greeting + '</div></div></div>' +
            '<div class="cw-inp-area"><input class="cw-inp" id="cw-inp" placeholder="Ask a question..." onkeypress="if(event.key===\'Enter\')sendMsg()"><button class="cw-send" onclick="sendMsg()">Send</button></div>' +
            '</div>';
        document.body.appendChild(widget);
        document.getElementById('cw-btn').addEventListener('click', () => {
            document.getElementById('cw-win').classList.add('open');
            document.getElementById('cw-btn').classList.add('hidden');
        });
    }

    function addMsg(text, role, sources) {
        const div = document.createElement('div');
        div.className = 'cw-msg ' + role;
        div.innerHTML = '<div class="cw-av">' + (role === 'user' ? '👤' : botAvatar) + '</div><div class="cw-txt">' + text + (sources && sources.length ? '<br><small style="opacity:0.7">📄 ' + [...new Set(sources.map(s=>s.document))].join(', ') + '</small>' : '') + '</div>';
        document.getElementById('cw-msgs').appendChild(div);
        document.getElementById('cw-msgs').scrollTop = document.getElementById('cw-msgs').scrollHeight;
    }

    async function sendMsg() {
        const inp = document.getElementById('cw-inp');
        const q = inp.value.trim();
        if (!q || isLoading) return;
        isLoading = true;
        addMsg(q, 'user');
        inp.value = '';
        try {
            const res = await fetch(apiUrl + '/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q, session_id: sessionId})
            });
            const data = await res.json();
            addMsg(data.answer || 'Error', 'bot', data.sources);
        } catch(e) {
            addMsg('Connection error', 'bot');
        }
        isLoading = false;
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', createWidget);
    else createWidget();
})();
'''

# ============================================================
# ROUTES
# ============================================================

@app.route("/api/index")
@app.route("/")
def index():
    """Main chat interface."""
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Knowledge Bot</title><meta name="viewport" content="width=device-width, initial-scale=1"></head>
    <body style="margin:0;font-family:sans-serif;">
        <h1 style="text-align:center;padding:20px;">🧠 Knowledge Bot</h1>
        <p style="text-align:center;">Click the button in the bottom-right corner to chat!</p>
        <script src="/widget.js"></script>
    </body>
    </html>
    '''

@app.route("/api/widget")
@app.route("/widget.js")
def serve_widget():
    """Serve the widget."""
    return Response(WIDGET_JS, mimetype='application/javascript')

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "Question required"}), 400

    try:
        docs = brain.intelligent_search(question, k=4)
        context = format_docs(docs)

        chain = QA_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        sources = []
        for doc in docs:
            src = doc.metadata.get("source_file", "?")
            if src not in [s["document"] for s in sources]:
                sources.append({"document": src, "page": doc.metadata.get("page_number", "?")})

        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def stats():
    return jsonify(brain.get_stats())

# ============================================================
# VERCEL HANDLER
# ============================================================
def handler(request, context):
    """Vercel serverless handler."""
    from flask import Request
    with app.request_context(Request(request.get("httpMethod", "GET"), request.get("headers", {}), request.get("body", ""))):
        return app.full_dispatch_request()