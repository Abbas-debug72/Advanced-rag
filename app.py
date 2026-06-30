# app.py – Pinecone RAG Chatbot with full debug logging
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

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Manual CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

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

# ------------------------------------------------------------------
# SIMPLE PING ENDPOINT
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# WIDGET ROUTE
# ------------------------------------------------------------------
@app.route('/widget.js')
def serve_widget():
    """Serve the widget JavaScript file"""
    try:
        return send_from_directory('.', 'widget.js')
    except Exception as e:
        print(f"⚠️ Error serving widget.js: {e}")
        return "Widget not found", 404
    """Serve the widget JavaScript file"""
    print("📨 Serving widget.js")
    try:
        return send_from_directory('.', 'widget.js')
    except:
        print("⚠️ widget.js not found, serving fallback")
        return """// Chat Widget - Knowledge Brain
(function() {
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
        // ... widget HTML (same as before)
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
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: sessionId })
            });
            
            console.log('📨 Response status:', res.status);
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
            addMessage('Sorry, an error occurred. Please try again.', 'bot');
        }
        isLoading = false;
    }

    // ... rest of widget functions
})();""", 200, {'Content-Type': 'application/javascript'}

# ------------------------------------------------------------------
# IMPROVED PROMPT
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Focus management
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
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
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
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
    print("\n🚀 Pinecone RAG Chatbot: http://127.0.0.1:5000")
    print("🧪 Test endpoint: /api/ping")
    app.run(debug=False, host="0.0.0.0", port=5000)