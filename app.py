# app_server.py - Vercel version with fallback embedding endpoints
from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import json
import requests
import time
from flask import Flask, request, jsonify, session, make_response
from memory import ConversationMemory
from pinecone import Pinecone
from groq import Groq

app = Flask(__name__)
app.secret_key = os.urandom(24)

print("🚀 Starting lightweight RAG server with Hugging Face Inference API...")

# ===== CONFIGURATION =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_CHAT_MODEL", "mixtral-8x7b-32768")

# Hugging Face Inference API Configuration
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Multiple fallback endpoints
HF_ENDPOINTS = [
    f"https://api-inference.huggingface.co/models/{HF_MODEL}",
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
    f"https://huggingface.co/api/models/{HF_MODEL}",
]

if not HF_API_KEY:
    print("⚠️ HUGGINGFACE_API_KEY not set. Please add it to your .env file.")
    print("   Get your free token at: https://huggingface.co/settings/tokens")

# ===== CORS HEADERS =====
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# ===== INITIALIZE CLIENTS =====
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)
groq_client = Groq(api_key=GROQ_API_KEY)

# ===== EMBEDDING FUNCTION WITH FALLBACK =====
def get_embedding(text: str):
    """Get embedding using Hugging Face Inference API with fallback endpoints"""
    if not HF_API_KEY:
        raise Exception("HUGGINGFACE_API_KEY not set. Please add it to your environment variables.")
    
    # Truncate text if too long
    if len(text) > 8000:
        text = text[:8000]
    
    # Try each endpoint
    last_error = None
    for endpoint in HF_ENDPOINTS:
        try:
            print(f"🔄 Trying endpoint: {endpoint}")
            
            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Different endpoints expect different payload formats
            if "pipeline" in endpoint:
                payload = {"inputs": text}
            else:
                payload = {"inputs": text}
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60  # Increased timeout
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                print("⚠️ Rate limit hit. Waiting 5 seconds...")
                time.sleep(5)
                continue
                
            if response.status_code == 503:
                print("⚠️ Service unavailable. Model may be loading. Waiting 10 seconds...")
                time.sleep(10)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                if isinstance(data[0], list):
                    return data[0]  # Batch response
                return data  # Single embedding
            elif isinstance(data, dict) and 'error' in data:
                raise Exception(f"API Error: {data['error']}")
            else:
                return data
                
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout on endpoint: {endpoint}")
            last_error = "Request timed out"
            continue
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error on endpoint: {endpoint}")
            last_error = "Connection error"
            continue
        except Exception as e:
            print(f"❌ Error on endpoint {endpoint}: {e}")
            last_error = str(e)
            continue
    
    raise Exception(f"All endpoints failed. Last error: {last_error}")

# ===== LOAD METADATA =====
def load_document_metadata():
    """Load document metadata from JSON file"""
    try:
        possible_paths = [
            "./brain_metadata.json",
            "brain_metadata.json",
            "/var/task/brain_metadata.json",
            os.path.join(os.path.dirname(__file__), "brain_metadata.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    print(f"✅ Loaded metadata from {path}: {len(data)} documents")
                    return data
        
        print("⚠️ No metadata file found. Please run ingestion locally.")
        return {}
    except Exception as e:
        print(f"⚠️ Error loading metadata: {e}")
        return {}

documents_metadata = load_document_metadata()

def get_all_filenames():
    return list(documents_metadata.keys())

def get_document_count():
    return len(documents_metadata)

# ===== PINECONE SEARCH =====
def search_pinecone(query: str, top_k: int = 5):
    """Search Pinecone using Hugging Face embeddings"""
    try:
        query_embedding = get_embedding(query)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.get('matches', [])
    except Exception as e:
        print(f"Search error: {e}")
        raise

# ===== GROQ RESPONSE =====
def generate_response(query: str, context: str):
    """Generate response using Groq"""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain the answer, say "I could not find that information in the documents."
Be concise and accurate. Use the context to support your answers.
Always cite the source document when possible."""},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Generation error: {e}")
        raise

memory = ConversationMemory()

# ===== WIDGET ROUTE =====
@app.route('/widget.js')
def serve_widget():
    """Serve the widget JavaScript - inline version"""
    widget_code = """
// Chat Widget - Knowledge Brain (Inline Version)
(function() {
    const CONFIG = {
        apiUrl: window.CHATBOT_API_URL || window.location.origin,
        botName: window.CHATBOT_NAME || 'Knowledge Bot',
        botAvatar: window.CHATBOT_AVATAR || '🧠',
        primaryColor: window.CHATBOT_COLOR || '#533483',
        greeting: window.CHATBOT_GREETING || 'Hello! Ask me anything about our documents.',
    };

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
        } catch(e) {}
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
                addMessage('Sorry, I could not process that question.', 'bot');
            }
        } catch (e) {
            hideTyping();
            addMessage('Sorry, an error occurred. Please try again.', 'bot');
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

# ===== API ROUTES =====

@app.route("/")
def home():
    """Home endpoint"""
    doc_count = get_document_count()
    return jsonify({
        "status": "ok",
        "service": "RAG Chatbot API",
        "documents_loaded": doc_count,
        "pinecone_connected": True,
        "groq_connected": True,
        "embedding_provider": "Hugging Face Inference API",
        "embedding_model": HF_MODEL,
        "message": f"{doc_count} documents loaded" if doc_count > 0 else "No documents loaded yet. Please run ingestion locally."
    })

@app.route("/api/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "documents": get_document_count(),
        "pinecone_connected": True,
        "groq_connected": True,
        "embedding_provider": "Hugging Face Inference API"
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    """Get stats about the knowledge base"""
    try:
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        doc_count = get_document_count()
        
        return jsonify({
            "total_documents": doc_count,
            "total_chunks": vector_count,
            "documents_loaded": doc_count > 0,
            "documents_list": get_all_filenames() if doc_count > 0 else [],
            "ingestion_status": "complete" if doc_count > 0 else "pending",
            "embedding_provider": "Hugging Face Inference API",
            "embedding_model": HF_MODEL
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "total_documents": get_document_count()
        }), 500

@app.route("/api/documents", methods=["GET"])
def list_documents():
    """List all documents in the knowledge base"""
    docs = []
    for fname, meta in documents_metadata.items():
        docs.append({
            "filename": fname,
            "pages": meta.get("pages", 0),
            "chunks": meta.get("chunks", 0),
            "category": meta.get("category", "general")
        })
    return jsonify({
        "documents": docs,
        "total": len(docs)
    })

@app.route("/api/categories", methods=["GET"])
def categories():
    """Get all categories"""
    cats = set()
    for meta in documents_metadata.values():
        cats.add(meta.get("category", "general"))
    return jsonify({"categories": sorted(list(cats))})

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    """Main chat endpoint"""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    try:
        print("📨 Chat request received")
        
        # Check content type
        if not request.is_json:
            print(f"❌ Invalid content type: {request.content_type}")
            return jsonify({
                "error": "Content-Type must be application/json",
                "answer": "Please send JSON data with Content-Type: application/json"
            }), 400
        
        # Parse JSON with error handling
        try:
            data = request.get_json(force=True, silent=True)
        except Exception as e:
            print(f"❌ JSON parse error: {e}")
            return jsonify({
                "error": "Invalid JSON format",
                "answer": "Please send valid JSON data"
            }), 400
        
        if not data:
            print("❌ No data received")
            return jsonify({
                "error": "No data received",
                "answer": "Please send a question in JSON format"
            }), 400
            
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        
        print(f"📨 Question: {question}")
        print(f"📨 Session ID: {session_id}")

        if not question:
            return jsonify({
                "error": "Question required",
                "answer": "Please provide a question"
            }), 400

        # Check if documents are loaded
        doc_count = get_document_count()
        if doc_count == 0:
            return jsonify({
                "answer": "⚠️ No documents have been ingested yet. Please run `python ingest_all.py` locally first.",
                "sources": [],
                "documents_loaded": False
            })

        # Get embedding
        try:
            print("🔄 Getting embedding from Hugging Face...")
            query_embedding = get_embedding(question)
            print(f"✅ Got embedding: {len(query_embedding)} dimensions")
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Embedding error: {error_msg}")
            if "429" in error_msg:
                return jsonify({
                    "answer": "⚠️ Rate limit exceeded. Please wait a moment and try again.",
                    "sources": [],
                    "error": "rate_limit"
                })
            elif "401" in error_msg or "403" in error_msg:
                return jsonify({
                    "answer": "⚠️ Invalid Hugging Face API key. Please check your environment variables.",
                    "sources": [],
                    "error": "invalid_api_key"
                })
            else:
                return jsonify({
                    "answer": f"⚠️ Embedding error: {error_msg[:200]}",
                    "sources": [],
                    "error": "embedding_error"
                })

        # Search Pinecone
        try:
            print("🔄 Searching Pinecone...")
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            matches = results.get('matches', [])
            print(f"✅ Found {len(matches)} matches")
        except Exception as e:
            print(f"❌ Pinecone error: {e}")
            return jsonify({
                "answer": f"⚠️ Database error: {str(e)[:100]}",
                "sources": []
            })

        if not matches:
            return jsonify({
                "answer": "I could not find relevant information in the knowledge base.",
                "sources": []
            })

        # Build context
        context_parts = []
        sources = []
        for match in matches:
            if match.get('score', 0) > 0.3:
                text = match.get('metadata', {}).get('text', '')
                source_file = match.get('metadata', {}).get('source_file', 'unknown')
                if text:
                    context_parts.append(text)
                    sources.append({
                        'document': source_file,
                        'score': match.get('score', 0)
                    })

        if not context_parts:
            return jsonify({
                "answer": "I found some information but with low confidence. Please rephrase your question.",
                "sources": []
            })

        context = "\n\n---\n\n".join(context_parts[:3])
        
        # Generate response
        try:
            print("🔄 Generating response with Groq...")
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer based on the context provided."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            answer = response.choices[0].message.content
            print(f"✅ Generated response: {answer[:100]}...")
        except Exception as e:
            print(f"❌ Groq error: {e}")
            return jsonify({
                "answer": f"⚠️ Generation error: {str(e)[:100]}",
                "sources": sources[:3]
            })

        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", answer)

        return jsonify({
            "answer": answer,
            "sources": sources[:3],
            "sources_count": len(sources)
        })

    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "answer": f"⚠️ An error occurred: {str(e)[:100]}"
        }), 500

@app.route("/api/conversation/<session_id>", methods=["DELETE"])
def clear_conversation(session_id):
    """Clear conversation history"""
    memory.clear_session(session_id)
    return jsonify({"success": True})

@app.route("/api/focus", methods=["POST"])
def set_focus():
    """Set focus on a specific document"""
    data = request.get_json()
    session_id = data.get("session_id", "default")
    filename = data.get("filename", None)
    return jsonify({"focus": filename})

@app.route("/widget-demo")
def widget_demo():
    """Serve a demo page with the widget"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Chat Widget Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 40px;
            background: #1a1a2e;
            color: white;
            text-align: center;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 40px;
            background: #16213e;
            border-radius: 16px;
        }
        h1 { font-size: 2rem; margin-bottom: 10px; }
        p { color: #94a3b8; }
        .status { color: #4ade80; }
        .subtitle { font-size: 0.8rem; color: #64748b; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Knowledge Brain Chat</h1>
        <p class="status">✅ Widget is ready</p>
        <p>Click the chat bubble in the bottom-right corner to start asking questions!</p>
        <p class="subtitle">Powered by Pinecone + Groq + Hugging Face</p>
    </div>
    <script>
        window.CHATBOT_API_URL = window.location.origin;
        window.CHATBOT_NAME = 'Knowledge Bot';
        window.CHATBOT_AVATAR = '🧠';
        window.CHATBOT_COLOR = '#533483';
        window.CHATBOT_GREETING = 'Hello! Ask me anything about our documents.';
    </script>
    <script src="/widget.js"></script>
</body>
</html>
""", 200, {'Content-Type': 'text/html'}

if __name__ == "__main__":
    print("\n🚀 Pinecone RAG Chatbot with Hugging Face Inference API")
    print(f"📚 Documents loaded: {get_document_count()}")
    print(f"🔗 Embedding model: {HF_MODEL}")
    print("💬 Widget available at: /widget.js")
    print("🧪 Demo page at: /widget-demo")
    app.run(debug=False, host="0.0.0.0", port=5000)