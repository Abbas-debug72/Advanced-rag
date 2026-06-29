"""
RAG Chatbot with Groq API, Pinecone, and Auto-Ingestion
Deployable on Vercel - USING GROQ FOR EMBEDDINGS
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq
from typing import List, Dict, Any
from functools import lru_cache
import time

# Force flush for Vercel logs
sys.stdout.flush()

print("🚀 Starting Flask app...", flush=True)

# ===== FORCE LOAD ENVIRONMENT VARIABLES =====
load_dotenv()
load_dotenv('.env.production')
load_dotenv('/var/task/.env')

print("📌 Checking environment variables...", flush=True)
print(f"   PINECONE_API_KEY: {'✅ SET' if os.getenv('PINECONE_API_KEY') else '❌ MISSING'}", flush=True)
print(f"   PINECONE_INDEX_HOST: {'✅ SET' if os.getenv('PINECONE_INDEX_HOST') else '❌ MISSING'}", flush=True)
print(f"   GROQ_API_KEY: {'✅ SET' if os.getenv('GROQ_API_KEY') else '❌ MISSING'}", flush=True)

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== FLASK APP =====
app = Flask(__name__)

# Manual CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/<path:path>', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def handle_options(path=None):
    return '', 200

# ===== CONFIGURATION =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain-groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "mixtral-8x7b-32768")
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small")

print(f"🔧 FINAL CONFIG:", flush=True)
print(f"   PINECONE_API_KEY: {'✅ SET' if PINECONE_API_KEY else '❌ MISSING'}", flush=True)
print(f"   PINECONE_INDEX_HOST: {PINECONE_INDEX_HOST}", flush=True)
print(f"   PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}", flush=True)
print(f"   GROQ_API_KEY: {'✅ SET' if GROQ_API_KEY else '❌ MISSING'}", flush=True)
print(f"   GROQ_EMBEDDING_MODEL: {GROQ_EMBEDDING_MODEL}", flush=True)

# ===== GLOBAL STATE =====
_pinecone_index = None
_groq_client = None

# ===== INITIALIZE CLIENTS =====
def initialize_clients():
    """Initialize Groq and Pinecone clients with better error handling"""
    global _pinecone_index, _groq_client
    
    print("=" * 60, flush=True)
    print("🔧 INITIALIZING CLIENTS...", flush=True)
    print("=" * 60, flush=True)
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_host = os.getenv("PINECONE_INDEX_HOST")
    groq_key = os.getenv("GROQ_API_KEY")
    
    print(f"📌 PINECONE_API_KEY: {'✅ SET' if pinecone_key else '❌ MISSING'}", flush=True)
    print(f"📌 PINECONE_INDEX_HOST: {'✅ SET' if pinecone_host else '❌ MISSING'}", flush=True)
    print(f"📌 GROQ_API_KEY: {'✅ SET' if groq_key else '❌ MISSING'}", flush=True)
    
    # Initialize Groq
    try:
        if groq_key:
            _groq_client = Groq(api_key=groq_key)
            print("✅ Groq client initialized successfully", flush=True)
            
            # Test Groq connection
            try:
                test_response = _groq_client.chat.completions.create(
                    model=GROQ_CHAT_MODEL,
                    messages=[{"role": "user", "content": "Say 'Hello'"}],
                    max_tokens=5
                )
                print("✅ Groq API test successful", flush=True)
            except Exception as e:
                print(f"⚠️ Groq API test failed: {e}", flush=True)
        else:
            print("❌ GROQ_API_KEY not set", flush=True)
            _groq_client = None
    except Exception as e:
        print(f"❌ Groq initialization error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        _groq_client = None
    
    # Initialize Pinecone
    try:
        if pinecone_key and pinecone_host:
            print("🔄 Connecting to Pinecone...", flush=True)
            print(f"   Host: {pinecone_host}", flush=True)
            pc = Pinecone(api_key=pinecone_key)
            _pinecone_index = pc.Index(host=pinecone_host)
            print(f"✅ Pinecone index connected successfully", flush=True)
            
            try:
                stats = _pinecone_index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                print(f"   📊 Vector count: {vector_count}", flush=True)
                
                if vector_count == 0:
                    print("   ⚠️ No vectors found in index", flush=True)
                else:
                    print(f"   ✅ {vector_count} vectors loaded", flush=True)
            except Exception as e:
                print(f"   ⚠️ Could not get index stats: {e}", flush=True)
        else:
            print("❌ PINECONE_API_KEY or PINECONE_INDEX_HOST not set", flush=True)
            _pinecone_index = None
            
    except Exception as e:
        print(f"❌ Pinecone initialization error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        _pinecone_index = None
    
    print("=" * 60, flush=True)
    print("✅ INITIALIZATION COMPLETE", flush=True)
    print(f"   Pinecone: {'✅ Connected' if _pinecone_index else '❌ Failed'}", flush=True)
    print(f"   Groq: {'✅ Connected' if _groq_client else '❌ Failed'}", flush=True)
    print("=" * 60, flush=True)

# ===== EMBEDDING FUNCTION USING GROQ =====
@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    """Get embedding using Groq API"""
    global _groq_client
    
    if _groq_client is None:
        raise Exception("Groq client not initialized")
    
    try:
        # Groq embeddings API call
        response = _groq_client.embeddings.create(
            model=GROQ_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def query_pinecone(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query Pinecone for relevant documents"""
    global _pinecone_index
    
    if _pinecone_index is None:
        raise Exception("Pinecone index not available")
    
    try:
        # Get query embedding using Groq
        query_embedding = get_embedding(query)
        
        results = _pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.get('matches', [])
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise

def generate_response(query: str, context: str) -> str:
    """Generate a response using Groq API"""
    global _groq_client
    
    if _groq_client is None:
        raise Exception("Groq client not initialized")
    
    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
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
        logger.error(f"Chat generation error: {e}")
        raise

def check_documents_exist() -> Dict[str, Any]:
    """Check if documents exist in Pinecone"""
    global _pinecone_index
    
    try:
        if _pinecone_index is None:
            return {
                'has_documents': False,
                'vector_count': 0,
                'status': 'error',
                'message': 'Pinecone not connected'
            }
        
        stats = _pinecone_index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        return {
            'has_documents': vector_count > 0,
            'vector_count': vector_count,
            'status': 'ready' if vector_count > 0 else 'empty',
            'message': '✅ Documents loaded' if vector_count > 0 else '❌ No documents found'
        }
    except Exception as e:
        logger.error(f"Error checking documents: {e}")
        return {
            'has_documents': False,
            'vector_count': 0,
            'status': 'error',
            'message': f'Error: {str(e)}'
        }

# ===== SERVE STATIC FILES =====
@app.route('/widget.js')
def serve_widget():
    """Serve the widget JavaScript file"""
    try:
        widget_path = os.path.join(os.path.dirname(__file__), 'widget.js')
        if os.path.exists(widget_path):
            with open(widget_path, 'r') as f:
                content = f.read()
            return content, 200, {'Content-Type': 'application/javascript'}
        else:
            # Return inline widget code if file doesn't exist
            return """// Chat Widget - Inline Version
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
})();""", 200, {'Content-Type': 'application/javascript'}
    except Exception as e:
        logger.error(f"Error serving widget: {e}")
        return f"Error: {str(e)}", 500

@app.route('/widget-demo')
def widget_demo():
    """Serve the widget demo page"""
    return """
<!DOCTYPE html>
<html>
<head><title>Chat Widget Demo</title></head>
<body style="font-family: Arial; padding: 40px; background: #1a1a2e; color: white; text-align: center;">
    <h1>🧠 Chat Widget Demo</h1>
    <p>The chat widget should appear in the bottom-right corner.</p>
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

# ===== API ROUTES =====

@app.route('/')
def home():
    """Serve the main HTML page"""
    doc_status = check_documents_exist()
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Brain Chatbot</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: #1a1a2e;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            flex-direction: column;
        }}
        .container {{
            text-align: center;
            padding: 40px;
            background: #16213e;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            max-width: 600px;
        }}
        h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .status {{ color: #4ade80; font-size: 1.2rem; }}
        .stats {{ color: #94a3b8; margin: 20px 0; }}
        .feature {{ background: #1a1a3e; padding: 12px; border-radius: 8px; margin: 8px 0; }}
        .badge {{
            display: inline-block;
            background: #533483;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            margin: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Knowledge Brain</h1>
        <p class="status">✅ All Systems Operational</p>
        <div class="stats">
            <div class="feature">📚 {doc_status.get('vector_count', 0)} Document Chunks Loaded</div>
            <div class="feature">🔗 Pinecone Vector Database Connected</div>
            <div class="feature">🤖 Groq LLM Ready</div>
            <div class="feature">📊 Using Groq for Embeddings</div>
        </div>
        <p style="color: #94a3b8; margin-top: 20px;">
            Click the chat bubble in the bottom-right corner to start asking questions!
        </p>
        <div>
            <span class="badge">Powered by Pinecone</span>
            <span class="badge">Groq AI</span>
            <span class="badge">Flask</span>
        </div>
    </div>
    <script>
        window.CHATBOT_API_URL = window.location.origin;
        window.CHATBOT_NAME = 'Knowledge Bot';
        window.CHATBOT_AVATAR = '🧠';
        window.CHATBOT_COLOR = '#533483';
        window.CHATBOT_GREETING = 'Hello! Ask me anything about our knowledge base.';
    </script>
    <script src="/widget.js"></script>
</body>
</html>
"""

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    doc_status = check_documents_exist()
    
    return jsonify({
        'status': 'healthy' if (_pinecone_index and _groq_client) else 'degraded',
        'pinecone_connected': _pinecone_index is not None,
        'groq_connected': _groq_client is not None,
        'documents_loaded': doc_status['has_documents'],
        'vector_count': doc_status['vector_count']
    })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check environment variables and connections"""
    env_status = {
        'PINECONE_API_KEY': {
            'exists': bool(os.getenv('PINECONE_API_KEY')),
            'length': len(os.getenv('PINECONE_API_KEY', '')) if os.getenv('PINECONE_API_KEY') else 0,
        },
        'PINECONE_INDEX_HOST': {
            'exists': bool(os.getenv('PINECONE_INDEX_HOST')),
            'value': os.getenv('PINECONE_INDEX_HOST', 'NOT SET')
        },
        'PINECONE_INDEX_NAME': {
            'value': os.getenv('PINECONE_INDEX_NAME', 'knowledge-brain-groq')
        },
        'GROQ_API_KEY': {
            'exists': bool(os.getenv('GROQ_API_KEY')),
            'length': len(os.getenv('GROQ_API_KEY', '')) if os.getenv('GROQ_API_KEY') else 0,
        },
        'GROQ_EMBEDDING_MODEL': {
            'value': os.getenv('GROQ_EMBEDDING_MODEL', 'text-embedding-3-small')
        },
        'GROQ_CHAT_MODEL': {
            'value': os.getenv('GROQ_CHAT_MODEL', 'mixtral-8x7b-32768')
        }
    }
    
    pinecone_status = 'NOT INITIALIZED'
    groq_status = 'NOT INITIALIZED'
    
    global _pinecone_index, _groq_client
    
    if _pinecone_index is not None:
        try:
            stats = _pinecone_index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            pinecone_status = f'CONNECTED - {vector_count} vectors'
        except Exception as e:
            pinecone_status = f'ERROR: {str(e)[:100]}'
    else:
        pinecone_status = 'DISCONNECTED'
    
    if _groq_client is not None:
        try:
            test_response = _groq_client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            groq_status = 'CONNECTED - Test passed'
        except Exception as e:
            groq_status = f'CONNECTED but test failed: {str(e)[:100]}'
    else:
        groq_status = 'DISCONNECTED'
    
    return jsonify({
        'environment_variables': env_status,
        'pinecone_status': pinecone_status,
        'groq_status': groq_status,
        'has_pinecone_key': bool(os.getenv('PINECONE_API_KEY')),
        'has_groq_key': bool(os.getenv('GROQ_API_KEY')),
        'has_pinecone_host': bool(os.getenv('PINECONE_INDEX_HOST'))
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get stats about the knowledge base"""
    try:
        if _pinecone_index is None:
            return jsonify({
                'total_documents': 0,
                'total_chunks': 0,
                'connected': False,
                'error': 'Pinecone not connected'
            })
        
        stats = _pinecone_index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        return jsonify({
            'total_documents': 10,
            'total_chunks': vector_count,
            'connected': True,
            'vector_count': vector_count,
            'index_name': PINECONE_INDEX_NAME,
            'embedding_model': GROQ_EMBEDDING_MODEL,
            'chat_model': GROQ_CHAT_MODEL
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'connected': False
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all categories"""
    return jsonify({
        'categories': ['general', 'academic', 'medical', 'resume/cv', 'guide']
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing "question" field'}), 400
        
        user_message = data['question'].strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        if _pinecone_index is None:
            return jsonify({
                'answer': "⚠️ Pinecone is not available. Please check your configuration.",
                'sources': [],
                'error': 'pinecone_not_connected'
            })
        
        if _groq_client is None:
            return jsonify({
                'answer': "⚠️ Groq is not available. Please check your configuration.",
                'sources': [],
                'error': 'groq_not_connected'
            })
        
        doc_status = check_documents_exist()
        
        if not doc_status['has_documents']:
            return jsonify({
                'answer': "⚠️ No documents have been ingested yet. Please run ingestion first.",
                'sources': [],
                'document_status': doc_status
            })
        
        try:
            matches = query_pinecone(user_message, top_k=5)
        except Exception as e:
            logger.error(f"Query error: {e}")
            return jsonify({
                'answer': f"Error querying knowledge base: {str(e)}",
                'sources': [],
                'error': str(e)
            }), 500
        
        if not matches:
            return jsonify({
                'answer': "I could not find any relevant information in the documents for your question.",
                'sources': []
            })
        
        context_parts = []
        sources = []
        
        for match in matches:
            if match.get('score', 0) > 0.3:
                text = match.get('metadata', {}).get('text', '')
                source_file = match.get('metadata', {}).get('source_file', 'unknown')
                page = match.get('metadata', {}).get('page_number', 1)
                if text:
                    context_parts.append(text)
                    sources.append({
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'score': match.get('score', 0),
                        'document': source_file,
                        'page': page
                    })
        
        if not context_parts:
            return jsonify({
                'answer': "I found some potentially relevant information, but none with high enough confidence. Please rephrase your question.",
                'sources': []
            })
        
        context = "\n\n---\n\n".join(context_parts[:3])
        
        try:
            response = generate_response(user_message, context)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return jsonify({
                'answer': f"Error generating response: {str(e)}",
                'sources': sources[:3]
            }), 500
        
        return jsonify({
            'answer': response,
            'sources': sources[:3],
            'sources_count': len(sources),
            'document_status': doc_status
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'answer': "An unexpected error occurred. Please try again."
        }), 500

# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ===== STARTUP =====
print("🔄 Initializing clients on module load...", flush=True)
initialize_clients()

if __name__ == '__main__':
    missing_vars = []
    if not PINECONE_API_KEY:
        missing_vars.append('PINECONE_API_KEY')
    if not PINECONE_INDEX_HOST:
        missing_vars.append('PINECONE_INDEX_HOST')
    if not GROQ_API_KEY:
        missing_vars.append('GROQ_API_KEY')
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}", flush=True)
    else:
        print("✅ All required environment variables are set", flush=True)
    
    doc_status = check_documents_exist()
    print(f"\n📊 Document status: {doc_status['status']}", flush=True)
    if doc_status['has_documents']:
        print(f"   ✅ {doc_status['vector_count']} vectors loaded", flush=True)
    else:
        print("   ⚠️ No documents found in Pinecone", flush=True)
    
    print("\n🚀 Starting Flask server on port 5000...", flush=True)
    print("=" * 60, flush=True)
    app.run(debug=False, host='0.0.0.0', port=5000)