# app.py – Working Widget Version
from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import re
import json
from flask import Flask, request, jsonify, Response

from brain import KnowledgeBrain
from memory import ConversationMemory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
app.secret_key = os.urandom(24)

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
# MAIN PAGE (with inline widget for testing)
# ============================================================
@app.route("/")
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Knowledge Bot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 { color: #1a1a2e; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🧠 Knowledge Bot</h1>
            <p>Click the chat button in the bottom-right corner to start asking questions!</p>
        </div>
        
        <script src="/widget.js"></script>
    </body>
    </html>
    '''

# ============================================================
# WIDGET JAVASCRIPT (Simplified & Tested)
# ============================================================
@app.route("/widget.js")
def serve_widget():
    widget_code = '''
(function() {
    // Create button
    var btn = document.createElement('button');
    btn.id = 'chatbot-btn';
    btn.innerHTML = '🧠';
    btn.style.cssText = 'position:fixed;bottom:20px;right:20px;width:60px;height:60px;border-radius:50%;background:#533483;color:white;border:none;cursor:pointer;font-size:28px;z-index:99999;box-shadow:0 4px 20px rgba(0,0,0,0.3);display:flex;align-items:center;justify-content:center;';
    document.body.appendChild(btn);
    
    // Create chat window
    var win = document.createElement('div');
    win.id = 'chatbot-win';
    win.style.cssText = 'position:fixed;bottom:90px;right:20px;width:380px;height:500px;background:#16213e;border-radius:16px;z-index:99999;display:none;flex-direction:column;overflow:hidden;font-family:sans-serif;box-shadow:0 8px 40px rgba(0,0,0,0.4);';
    win.innerHTML = '<div style="background:linear-gradient(135deg,#0f3460,#533483);color:white;padding:16px;font-weight:bold;display:flex;gap:10px;align-items:center;">🧠 Knowledge Bot<span style="margin-left:auto;cursor:pointer;font-size:18px;" onclick="document.getElementById(\\'chatbot-win\\').style.display=\\'none\\';document.getElementById(\\'chatbot-btn\\').style.display=\\'flex\\'">✕</span></div><div id="chatbot-msgs" style="flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;"><div style="display:flex;gap:8px;"><div style="width:30px;height:30px;border-radius:50%;background:#533483;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;">🧠</div><div style="padding:10px 14px;border-radius:12px;font-size:14px;color:white;background:#1a1a3e;">Hello! Ask me anything.</div></div></div><div style="display:flex;padding:12px;border-top:1px solid #1a1a3e;gap:8px;"><input id="chatbot-inp" placeholder="Ask a question..." style="flex:1;padding:10px;border:1px solid #2d2d5e;border-radius:20px;background:#1a1a3e;color:white;font-size:14px;outline:none;"><button onclick="sendChatMsg()" style="padding:10px 18px;background:#533483;color:white;border:none;border-radius:20px;cursor:pointer;">Send</button></div>';
    document.body.appendChild(win);
    
    // Toggle
    btn.onclick = function() {
        win.style.display = 'flex';
        btn.style.display = 'none';
        document.getElementById('chatbot-inp').focus();
    };
    
    var sessionId = 'session_' + Date.now();
    var loading = false;
    
    // Send message function
    window.sendChatMsg = function() {
        var inp = document.getElementById('chatbot-inp');
        var q = inp.value.trim();
        if (!q || loading) return;
        loading = true;
        
        // Add user message
        var msgs = document.getElementById('chatbot-msgs');
        msgs.innerHTML += '<div style="display:flex;gap:8px;align-self:flex-end;flex-direction:row-reverse;max-width:85%;"><div style="width:30px;height:30px;border-radius:50%;background:#0f3460;display:flex;align-items:center;justify-content:center;font-size:14px;">👤</div><div style="padding:10px 14px;border-radius:12px;font-size:14px;color:white;background:#533483;">' + q + '</div></div>';
        inp.value = '';
        msgs.scrollTop = msgs.scrollHeight;
        
        // Call API
        fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: q, session_id: sessionId})
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            msgs.innerHTML += '<div style="display:flex;gap:8px;max-width:85%;"><div style="width:30px;height:30px;border-radius:50%;background:#533483;display:flex;align-items:center;justify-content:center;font-size:14px;">🧠</div><div style="padding:10px 14px;border-radius:12px;font-size:14px;color:white;background:#1a1a3e;">' + (data.answer || 'Error') + '</div></div>';
            msgs.scrollTop = msgs.scrollHeight;
            loading = false;
        })
        .catch(function() {
            msgs.innerHTML += '<div style="display:flex;gap:8px;max-width:85%;"><div style="width:30px;height:30px;border-radius:50%;background:#533483;display:flex;align-items:center;justify-content:center;font-size:14px;">🧠</div><div style="padding:10px 14px;border-radius:12px;font-size:14px;color:white;background:#1a1a3e;">Connection error</div></div>';
            loading = false;
        });
    };
    
    // Enter key
    document.getElementById('chatbot-inp').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendChatMsg();
    });
})();
'''
    return Response(widget_code, mimetype='application/javascript')

# ============================================================
# API ROUTES
# ============================================================
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

if __name__ == "__main__":
    print("\n🚀 Running at http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)