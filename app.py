# app.py – Complete Working Version with Fixed Widget
from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from brain import KnowledgeBrain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)

print("🧠 Loading Knowledge Brain...")
brain = KnowledgeBrain(pdf_directory=os.getenv("PDF_DIRECTORY", "./pdfs"))

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024)

PROMPT = ChatPromptTemplate.from_template(
    "Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

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
# MAIN PAGE WITH CHAT INTERFACE
# ============================================================
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Knowledge Bot</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f5f5f5;
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .header h1 { color: #1a1a2e; font-size: 2rem; margin-bottom: 8px; }
            .header p { color: #718096; }
            .chatbox {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(135deg, #0f3460, #533483);
                color: white;
                padding: 16px 24px;
                font-weight: bold;
                font-size: 1.1rem;
            }
            .messages {
                height: 400px;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .msg {
                display: flex;
                gap: 10px;
                max-width: 80%;
            }
            .msg.user {
                align-self: flex-end;
                flex-direction: row-reverse;
            }
            .msg .avatar {
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }
            .msg.bot .avatar { background: #533483; color: white; }
            .msg.user .avatar { background: #0f3460; color: white; }
            .msg .text {
                padding: 10px 16px;
                border-radius: 12px;
                line-height: 1.5;
                font-size: 0.95rem;
            }
            .msg.bot .text { background: #e2e8f0; color: #1a1a2e; }
            .msg.user .text { background: #533483; color: white; }
            .input-area {
                display: flex;
                padding: 16px;
                border-top: 1px solid #e2e8f0;
                gap: 10px;
            }
            #question {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e2e8f0;
                border-radius: 25px;
                font-size: 0.95rem;
                outline: none;
                transition: border-color 0.2s;
            }
            #question:focus { border-color: #533483; }
            #send {
                padding: 12px 24px;
                background: #533483;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 0.95rem;
                font-weight: 600;
            }
            #send:hover { background: #6c4fa0; }
            .sources {
                margin-top: 4px;
                font-size: 0.7rem;
                color: #a0aec0;
            }
            .loading-text {
                color: #a0aec0;
                font-style: italic;
                padding: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Knowledge Bot</h1>
                <p>Ask questions about the documents in the knowledge base</p>
            </div>
            <div class="chatbox">
                <div class="chat-header">💬 Chat</div>
                <div class="messages" id="messages">
                    <div class="msg bot">
                        <div class="avatar">🧠</div>
                        <div class="text">Hello! Ask me anything about the knowledge base documents.</div>
                    </div>
                </div>
                <div class="input-area">
                    <input type="text" id="question" placeholder="Type your question..." onkeypress="if(event.key==='Enter')ask()" autofocus>
                    <button id="send" onclick="ask()">Send</button>
                </div>
            </div>
        </div>

        <script>
            var sessionId = 'session_' + Date.now();
            var loading = false;

            function addMessage(text, role, sources) {
                var div = document.createElement('div');
                div.className = 'msg ' + role;
                var avatarEmoji = role === 'user' ? '👤' : '🧠';
                var sourceHTML = sources && sources.length > 0 ? '<div class="sources">📄 ' + [...new Set(sources.map(function(s){return s.document}))].join(', ') + '</div>' : '';
                div.innerHTML = '<div class="avatar">' + avatarEmoji + '</div><div class="text">' + text.replace(/\\n/g, '<br>') + sourceHTML + '</div>';
                document.getElementById('messages').appendChild(div);
                document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
            }

            function showLoading() {
                var div = document.createElement('div');
                div.className = 'msg bot';
                div.id = 'loading';
                div.innerHTML = '<div class="avatar">🧠</div><div class="text"><div class="loading-text">Thinking...</div></div>';
                document.getElementById('messages').appendChild(div);
                document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
            }

            function hideLoading() {
                var el = document.getElementById('loading');
                if (el) el.remove();
            }

            async function ask() {
                var input = document.getElementById('question');
                var q = input.value.trim();
                if (!q || loading) return;
                
                loading = true;
                addMessage(q, 'user');
                input.value = '';
                showLoading();
                
                try {
                    var res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: q, session_id: sessionId})
                    });
                    var data = await res.json();
                    hideLoading();
                    addMessage(data.answer || 'Sorry, an error occurred.', 'bot', data.sources);
                } catch(e) {
                    hideLoading();
                    addMessage('Connection error. Please try again.', 'bot');
                }
                loading = false;
            }
        </script>
    </body>
    </html>
    """

# ============================================================
# WIDGET JAVASCRIPT (FIXED - Always calls your Vercel API)
# ============================================================
@app.route("/widget.js")
def widget():
    js = """
    (function(){
        // FIXED: Always use your Vercel URL, not the embedding site's URL
        var api = 'https://advanced-6jxkyhxli-gat6.vercel.app';
        
        var btn = document.createElement('button');
        btn.innerHTML = '🧠';
        btn.style.cssText = 'position:fixed;bottom:20px;right:20px;width:60px;height:60px;border-radius:50%;background:#533483;color:white;border:none;cursor:pointer;font-size:28px;z-index:99999;box-shadow:0 4px 20px rgba(0,0,0,0.3);display:flex;align-items:center;justify-content:center;';
        
        var win = document.createElement('div');
        win.style.cssText = 'position:fixed;bottom:90px;right:20px;width:380px;height:500px;background:#16213e;border-radius:16px;z-index:99999;display:none;flex-direction:column;overflow:hidden;font-family:sans-serif;box-shadow:0 8px 40px rgba(0,0,0,0.4);';
        
        win.innerHTML = '<div style="background:linear-gradient(135deg,#0f3460,#533483);color:white;padding:16px;font-weight:bold;display:flex;justify-content:space-between;align-items:center;">🧠 Knowledge Bot<span id="cw-close" style="cursor:pointer;font-size:18px;">✕</span></div>' +
            '<div id="cw-msgs" style="flex:1;overflow-y:auto;padding:16px;"><div style="color:white;font-size:14px;">Hello! Ask me anything.</div></div>' +
            '<div style="display:flex;padding:12px;border-top:1px solid #1a1a3e;gap:8px;">' +
            '<input id="cw-inp" placeholder="Ask a question..." style="flex:1;padding:10px;border:none;border-radius:20px;background:#1a1a3e;color:white;font-size:14px;outline:none;">' +
            '<button id="cw-send" style="padding:10px 18px;background:#533483;color:white;border:none;border-radius:20px;cursor:pointer;">Send</button></div>';
        
        document.body.appendChild(btn);
        document.body.appendChild(win);
        
        var session = 'ws_' + Date.now();
        var loading = false;
        
        btn.onclick = function(){
            win.style.display = 'flex';
            btn.style.display = 'none';
            document.getElementById('cw-inp').focus();
        };
        
        document.getElementById('cw-close').onclick = function(){
            win.style.display = 'none';
            btn.style.display = 'flex';
        };
        
        function addMsg(text, role) {
            var d = document.createElement('div');
            d.style.cssText = 'padding:8px 12px;margin:4px 0;border-radius:10px;font-size:13px;color:white;max-width:85%;word-wrap:break-word;' + (role==='user'?'background:#533483;margin-left:auto;':'background:#1a1a3e;');
            d.textContent = text;
            document.getElementById('cw-msgs').appendChild(d);
            document.getElementById('cw-msgs').scrollTop = document.getElementById('cw-msgs').scrollHeight;
        }
        
        async function send() {
            var inp = document.getElementById('cw-inp');
            var q = inp.value.trim();
            if(!q || loading) return;
            loading = true;
            addMsg(q, 'user');
            inp.value = '';
            try {
                var r = await fetch(api + '/api/chat', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({question:q, session_id:session})
                });
                var d = await r.json();
                addMsg(d.answer||'Error', 'bot');
            } catch(e) {
                addMsg('Connection error. Please try again.', 'bot');
            }
            loading = false;
        }
        
        document.getElementById('cw-send').onclick = send;
        document.getElementById('cw-inp').addEventListener('keypress', function(e){ if(e.key==='Enter') send(); });
        
        console.log('✅ Chat widget ready! API: ' + api);
    })();
    """
    return Response(js, mimetype='application/javascript')

# ============================================================
# API ROUTES
# ============================================================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    q = data.get("question", "").strip()
    if not q:
        return jsonify({"error": "Question required"}), 400
    try:
        docs = brain.intelligent_search(q, k=4)
        ctx = format_docs(docs)
        chain = PROMPT | llm | StrOutputParser()
        ans = chain.invoke({"context": ctx, "question": q})
        sources = []
        for d in docs:
            src = d.metadata.get("source_file", "?")
            if src not in [s["document"] for s in sources]:
                sources.append({"document": src, "page": d.metadata.get("page_number", "?")})
        return jsonify({"answer": ans, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def stats():
    return jsonify(brain.get_stats())

if __name__ == "__main__":
    print("\n🚀 Running at http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)