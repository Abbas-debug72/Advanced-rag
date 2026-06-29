from dotenv import load_dotenv
load_dotenv()

import os
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from brain import KnowledgeBrain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)

# Load brain ONCE at module level (Vercel caches this)
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
        llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    return llm

PROMPT = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def format_docs(docs):
    parts = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get('source_file', '?')
        if src in seen:
            continue
        seen.add(src)
        parts.append(f"[{src}]\n{doc.page_content[:300]}\n")
    return "\n".join(parts)

@app.route("/")
def index():
    return """<!DOCTYPE html><html><head><title>Knowledge Bot</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font-family:sans-serif;text-align:center;padding:50px;background:#f5f5f5}
h1{color:#1a1a2e} .btn{background:#533483;color:white;padding:16px 32px;border:none;border-radius:12px;font-size:18px;cursor:pointer;margin-top:20px}
</style></head><body><h1>🧠 Knowledge Bot</h1><p>Click below to open the chat widget</p>
<button class="btn" onclick="document.getElementById('w').style.display='block';this.style.display='none'">Open Chat</button>
<div id="w" style="display:none;max-width:600px;margin:20px auto;background:white;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.1);overflow:hidden">
<div style="background:#533483;color:white;padding:16px;font-weight:bold">💬 Chat</div>
<div id="msgs" style="height:350px;overflow-y:auto;padding:16px;text-align:left"><div style="color:#666">Hello! Ask me anything.</div></div>
<div style="display:flex;padding:12px;border-top:1px solid #eee;gap:8px"><input id="inp" placeholder="Ask a question..." style="flex:1;padding:12px;border:1px solid #ddd;border-radius:8px;font-size:14px;outline:none" onkeypress="if(event.key==='Enter')ask()"><button onclick="ask()" style="padding:12px 24px;background:#533483;color:white;border:none;border-radius:8px;cursor:pointer">Send</button></div></div>
<script>
var sid='s'+Date.now(),loading=false;
function add(t,r){var d=document.createElement('div');d.style.cssText='margin:8px 0;padding:10px 14px;border-radius:10px;max-width:80%;'+(r==='user'?'background:#533483;color:white;margin-left:auto;':'background:#f0f0f0;');d.textContent=t;document.getElementById('msgs').appendChild(d);document.getElementById('msgs').scrollTop=document.getElementById('msgs').scrollHeight}
async function ask(){var i=document.getElementById('inp'),q=i.value.trim();if(!q||loading)return;loading=true;add(q,'user');i.value='';try{var r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,session_id:sid})}),d=await r.json();add(d.answer||'Error','bot')}catch(e){add('Connection error','bot')}loading=false}
</script></body></html>"""

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        q = data.get("question", "").strip()
        if not q:
            return jsonify({"answer": "Please ask a question."})
        
        brain = get_brain()
        llm = get_llm()
        
        docs = brain.intelligent_search(q, k=3)
        ctx = format_docs(docs)
        
        chain = PROMPT | llm | StrOutputParser()
        ans = chain.invoke({"context": ctx, "question": q})
        
        sources = []
        for d in docs:
            src = d.metadata.get("source_file", "?")
            if src not in [s["document"] for s in sources]:
                sources.append({"document": src})
        
        return jsonify({"answer": ans, "sources": sources})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)[:200]}"})

@app.route("/api/stats")
def stats():
    try:
        brain = get_brain()
        return jsonify(brain.get_stats())
    except:
        return jsonify({"error": "Stats unavailable"})

# Widget for embedding
@app.route("/widget.js")
def widget():
    js = """(function(){var api='https://advanced-6jxkyhxli-gat6.vercel.app';var btn=document.createElement('button');btn.innerHTML='🧠';btn.style.cssText='position:fixed;bottom:20px;right:20px;width:60px;height:60px;border-radius:50%;background:#533483;color:white;border:none;cursor:pointer;font-size:28px;z-index:99999;box-shadow:0 4px 20px rgba(0,0,0,0.3)';var win=document.createElement('div');win.id='cw';win.innerHTML='<div style="background:#533483;color:white;padding:14px;font-weight:bold;display:flex;justify-content:space-between">🧠 Knowledge Bot<span onclick="document.getElementById(\\'cw\\').style.display=\\'none\\';document.getElementById(\\'cb\\').style.display=\\'block\\'" style="cursor:pointer">✕</span></div><div id="cm" style="height:360px;overflow-y:auto;padding:14px;font-size:14px"><div style="color:white">Hello! Ask me anything.</div></div><div style="display:flex;padding:10px;gap:8px"><input id="ci" placeholder="Ask a question..." style="flex:1;padding:10px;border:none;border-radius:20px;font-size:14px;outline:none"><button onclick="cs()" style="padding:10px 18px;background:#533483;color:white;border:none;border-radius:20px;cursor:pointer">Send</button></div>';win.style.cssText='position:fixed;bottom:90px;right:20px;width:360px;height:480px;background:#16213e;border-radius:16px;z-index:99999;display:none;flex-direction:column;overflow:hidden;font-family:sans-serif;box-shadow:0 8px 40px rgba(0,0,0,0.4);color:white';btn.id='cb';document.body.appendChild(btn);document.body.appendChild(win);var sid='w'+Date.now(),loading=false;btn.onclick=function(){win.style.display='flex';btn.style.display='none';document.getElementById('ci').focus()};window.cs=function(){var i=document.getElementById('ci'),q=i.value.trim();if(!q||loading)return;loading=true;var m=document.getElementById('cm');m.innerHTML+='<div style="text-align:right;margin:6px 0"><span style="background:#533483;padding:8px 12px;border-radius:12px;display:inline-block;max-width:80%">'+q+'</span></div>';i.value='';m.scrollTop=m.scrollHeight;var x=new XMLHttpRequest();x.open('POST',api+'/api/chat',true);x.setRequestHeader('Content-Type','application/json');x.onload=function(){loading=false;if(x.status===200){var d=JSON.parse(x.responseText);m.innerHTML+='<div style="margin:6px 0"><span style="background:#1a1a3e;padding:8px 12px;border-radius:12px;display:inline-block;max-width:80%">'+(d.answer||'No answer')+'</span></div>'}else{m.innerHTML+='<div style="color:#ff6b6b;margin:6px 0">Error</div>'}m.scrollTop=m.scrollHeight};x.onerror=function(){loading=false;m.innerHTML+='<div style="color:#ff6b6b;margin:6px 0">Connection error</div>'};x.send(JSON.stringify({question:q,session_id:sid}))};document.getElementById('ci').addEventListener('keypress',function(e){if(e.key==='Enter')cs()})})();"""
    return Response(js, mimetype='application/javascript')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)