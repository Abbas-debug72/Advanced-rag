from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify, Response

from brain import KnowledgeBrain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

# Manual CORS
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/api/chat', methods=['OPTIONS'])
def options():
    return '', 200

print("🧠 Loading Knowledge Brain...")
brain = None
llm = None

def get_brain():
    global brain
    if brain is None:
        brain = KnowledgeBrain(pdf_directory=os.getenv("PDF_DIRECTORY", "./pdfs"))
    return brain

def get_llm():
    global llm
    if llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024)
    return llm

# BALANCED PROMPT
PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant answering questions about documents.\n\n"
    "Use the context below to answer the question. If the context contains relevant "
    "information, use it. If the context clearly doesn't have the answer, say so.\n\n"
    "Context from documents:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

def format_docs(docs):
    parts = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get('source_file', '?')
        if src in seen: continue
        seen.add(src)
        parts.append(f"[Document: {src}]\n{doc.page_content[:600]}\n")
    return "\n".join(parts)

@app.route("/")
def index():
    return """<!DOCTYPE html><html><head><title>Knowledge Bot</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font-family:sans-serif;max-width:650px;margin:40px auto;padding:20px;background:#f5f5f5}
.card{background:white;padding:30px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,0.1)}
h1{color:#1a1a2e;text-align:center}
input{width:100%;padding:12px;border:2px solid #533483;border-radius:25px;font-size:15px;margin:10px 0;box-sizing:border-box;outline:none}
input:focus{border-color:#6c4fa0}
button{width:100%;padding:12px;background:#533483;color:white;border:none;border-radius:25px;font-size:15px;cursor:pointer;font-weight:bold}
button:hover{background:#6c4fa0}
#msgs{max-height:400px;overflow-y:auto;margin:15px 0}
.msg{margin:6px 0;padding:10px 14px;border-radius:12px;max-width:85%;line-height:1.4;font-size:14px}
.user{background:#533483;color:white;margin-left:auto}
.bot{background:#e2e8f0;color:#1a1a2e}
.doc{font-size:11px;color:#718096;margin-top:4px}
</style></head><body><div class="card">
<h1>🧠 Knowledge Bot</h1><div id="msgs"><div class="msg bot">Hello! Ask me anything about the documents.</div></div>
<input id="q" placeholder="Ask a question..." onkeypress="if(event.key==='Enter')ask()" autofocus><button onclick="ask()">Send</button>
</div><script>
var sid='s'+Date.now(),loading=false;
function add(t,r,s){var d=document.createElement('div');d.className='msg '+r;d.innerHTML=t+(s?'<div class="doc">'+s+'</div>':'');document.getElementById('msgs').appendChild(d);document.getElementById('msgs').scrollTop=document.getElementById('msgs').scrollHeight}
async function ask(){var i=document.getElementById('q'),q=i.value.trim();if(!q||loading)return;loading=true;add(q,'user');i.value='';try{var r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,session_id:sid})}),d=await r.json();var src=d.sources&&d.sources.length?'📄 '+d.sources.map(function(s){return s.document}).join(', '):'';add(d.answer||'Error','bot',src)}catch(e){add('Connection error','bot')}loading=false}
</script></body></html>"""

@app.route("/widget.js")
def widget():
    js = """(function(){var api='https://advanced-ki8zq3ubn-gat6.vercel.app';var btn=document.createElement('button');btn.innerHTML='🧠';btn.style.cssText='position:fixed;bottom:20px;right:20px;width:60px;height:60px;border-radius:50%;background:#533483;color:white;border:none;cursor:pointer;font-size:28px;z-index:99999;box-shadow:0 4px 20px rgba(0,0,0,0.3)';var win=document.createElement('div');win.id='cw';win.innerHTML='<div style="background:#533483;color:white;padding:14px;font-weight:bold;display:flex;justify-content:space-between">🧠 Knowledge Bot<span onclick="document.getElementById(\\'cw\\').style.display=\\'none\\';document.getElementById(\\'cb\\').style.display=\\'block\\'" style="cursor:pointer">✕</span></div><div id="cm" style="height:340px;overflow-y:auto;padding:14px;font-size:14px"><div style="color:white">Hello! Ask me anything about the documents.</div></div><div style="display:flex;padding:10px;gap:8px"><input id="ci" placeholder="Ask..." style="flex:1;padding:10px;border:none;border-radius:20px;font-size:14px;outline:none;color:white;background:#1a1a3e"><button onclick="cs()" style="padding:10px 18px;background:#533483;color:white;border:none;border-radius:20px;cursor:pointer;font-size:14px">Send</button></div>';win.style.cssText='position:fixed;bottom:90px;right:20px;width:370px;height:460px;background:#16213e;border-radius:16px;z-index:99999;display:none;flex-direction:column;overflow:hidden;font-family:sans-serif;box-shadow:0 8px 40px rgba(0,0,0,0.4);color:white';btn.id='cb';document.body.appendChild(btn);document.body.appendChild(win);var sid='w'+Date.now(),loading=false;btn.onclick=function(){win.style.display='flex';btn.style.display='none';document.getElementById('ci').focus()};window.cs=function(){var i=document.getElementById('ci'),q=i.value.trim();if(!q||loading)return;loading=true;var m=document.getElementById('cm');m.innerHTML+='<div style="text-align:right;margin:6px 0"><span style="background:#533483;padding:8px 12px;border-radius:12px;display:inline-block;max-width:80%">'+q+'</span></div>';i.value='';m.scrollTop=m.scrollHeight;var x=new XMLHttpRequest();x.open('POST',api+'/api/chat',true);x.setRequestHeader('Content-Type','application/json');x.onload=function(){loading=false;if(x.status===200){var d=JSON.parse(x.responseText);m.innerHTML+='<div style="margin:6px 0"><span style="background:#1a1a3e;padding:8px 12px;border-radius:12px;display:inline-block;max-width:80%">'+(d.answer||'No answer').replace(/\\n/g,'<br>')+'</span></div>'}else{m.innerHTML+='<div style="color:#ff6b6b;margin:6px 0">Error '+x.status+'</div>'}m.scrollTop=m.scrollHeight};x.onerror=function(){loading=false;m.innerHTML+='<div style="color:#ff6b6b;margin:6px 0">Connection error</div>'};x.send(JSON.stringify({question:q,session_id:sid}))};document.getElementById('ci').addEventListener('keypress',function(e){if(e.key==='Enter')cs()})})()"""
    return Response(js, mimetype='application/javascript')

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        q = data.get("question", "").strip()
        if not q:
            return jsonify({"answer": "Please ask a question.", "sources": []})
        
        b = get_brain()
        l = get_llm()
        
        # Get more chunks for better context
        docs = b.intelligent_search(q, k=5)
        ctx = format_docs(docs)
        
        chain = PROMPT | l | StrOutputParser()
        ans = chain.invoke({"context": ctx, "question": q})
        
        sources = []
        for d in docs:
            src = d.metadata.get("source_file", "?")
            if src not in [s["document"] for s in sources]:
                sources.append({"document": src, "page": d.metadata.get("page_number", "?")})
        
        return jsonify({"answer": ans, "sources": sources})
    except Exception as e:
        return jsonify({"answer": "Sorry, an error occurred. Please try again.", "sources": []})

@app.route("/api/stats")
def stats():
    try:
        return jsonify(get_brain().get_stats())
    except:
        return jsonify({"error": "Stats unavailable"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)