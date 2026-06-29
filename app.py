"""
RAG Chatbot with Groq API, Pinecone, and Auto-Ingestion
Deployable on Vercel
"""

import os
import sys
import subprocess
import threading
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pinecone
from groq import Groq
from typing import List, Dict, Any, Optional
from functools import lru_cache
import time

# Force flush for Vercel logs
sys.stdout.flush()

print("🚀 Starting Flask app...", flush=True)

# Load environment variables
load_dotenv()
print("✅ Environment variables loaded", flush=True)

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== FLASK APP (No CORS) =====
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
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "mixtral-8x7b-32768")
INGEST_ON_START = os.getenv("INGEST_ON_START", "true").lower() == "true"
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")

print(f"🔧 CONFIG:", flush=True)
print(f"   PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}", flush=True)
print(f"   GROQ_CHAT_MODEL: {GROQ_CHAT_MODEL}", flush=True)
print(f"   INGEST_ON_START: {INGEST_ON_START}", flush=True)
print(f"   DOCUMENTS_DIR: {DOCUMENTS_DIR}", flush=True)

# ===== GLOBAL STATE =====
_pinecone_index = None
_groq_client = None
_is_ingesting = False
_ingestion_complete = False
_ingestion_error = None

# ===== CHECK IF DOCUMENTS EXIST IN PINECONE =====
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

# ===== RUN INGEST_ALL.PY =====
def run_ingestion():
    """Run ingest_all.py in a separate process"""
    global _is_ingesting, _ingestion_complete, _ingestion_error
    
    if _is_ingesting:
        logger.info("Ingestion already in progress...")
        return
    
    _is_ingesting = True
    _ingestion_complete = False
    _ingestion_error = None
    
    def ingest_thread():
        global _is_ingesting, _ingestion_complete, _ingestion_error
        
        try:
            logger.info("📄 Starting ingestion of documents...")
            print("📄 Starting ingestion...", flush=True)
            
            # Check if ingest_all.py exists
            if not os.path.exists("ingest_all.py"):
                logger.error("ingest_all.py not found!")
                _ingestion_error = "ingest_all.py not found"
                _is_ingesting = False
                return
            
            # Run the Python script
            result = subprocess.run(
                ["python3", "ingest_all.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Ingestion completed successfully!")
                print("✅ Ingestion completed successfully!", flush=True)
                _ingestion_complete = True
            else:
                error_msg = f"❌ Ingestion failed with code {result.returncode}"
                logger.error(error_msg)
                logger.error(f"stderr: {result.stderr}")
                _ingestion_error = error_msg
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Ingestion timed out after 5 minutes")
            _ingestion_error = "Ingestion timed out"
        except Exception as e:
            logger.error(f"❌ Ingestion error: {e}")
            _ingestion_error = str(e)
        finally:
            _is_ingesting = False
    
    # Start in background thread
    thread = threading.Thread(target=ingest_thread)
    thread.daemon = True
    thread.start()
    logger.info("📄 Ingestion started in background thread")

# ===== INITIALIZE CLIENTS =====
def initialize_clients():
    """Initialize Groq and Pinecone clients"""
    global _pinecone_index, _groq_client
    
    try:
        # Groq client
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq client initialized")
        
        # Pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )
        logger.info("✅ Pinecone initialized")
        
        if PINECONE_INDEX_NAME in pinecone.list_indexes():
            _pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
            logger.info(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' exists")
            
            # Check if documents exist
            doc_status = check_documents_exist()
            
            if doc_status['has_documents']:
                logger.info(f"✅ Documents already loaded: {doc_status['vector_count']} vectors")
            else:
                logger.info("⚠️ No documents found in Pinecone")
                
                # Run ingestion automatically if enabled
                if INGEST_ON_START:
                    logger.info("🔄 Auto-ingestion enabled. Starting ingest_all.py...")
                    run_ingestion()
                else:
                    logger.info("ℹ️ Auto-ingestion disabled. Set INGEST_ON_START=true to enable")
        else:
            logger.warning(f"⚠️ Pinecone index '{PINECONE_INDEX_NAME}' does not exist")
            _pinecone_index = None
            
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        _pinecone_index = None
        _groq_client = None

# ===== GROQ FUNCTIONS =====
@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    """Get embedding using Groq API"""
    global _groq_client
    
    if _groq_client is None:
        raise Exception("Groq client not initialized")
    
    try:
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
    
    query_embedding = get_embedding(query)
    
    results = _pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results.get('matches', [])

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
Be concise and accurate. Use the context to support your answers."""},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Chat generation error: {e}")
        raise

# ===== API ROUTES =====

@app.route('/')
def home():
    """Home endpoint"""
    doc_status = check_documents_exist()
    
    return jsonify({
        'status': 'ok',
        'service': 'RAG Chatbot',
        'pinecone_connected': _pinecone_index is not None,
        'groq_connected': _groq_client is not None,
        'documents_loaded': doc_status['has_documents'],
        'vector_count': doc_status['vector_count'],
        'is_ingesting': _is_ingesting,
        'ingestion_complete': _ingestion_complete,
        'ingestion_error': _ingestion_error
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    doc_status = check_documents_exist()
    
    return jsonify({
        'status': 'healthy',
        'pinecone_connected': _pinecone_index is not None,
        'groq_connected': _groq_client is not None,
        'documents_loaded': doc_status['has_documents'],
        'vector_count': doc_status['vector_count'],
        'is_ingesting': _is_ingesting,
        'ingestion_complete': _ingestion_complete
    })

@app.route('/api/ingest-check', methods=['GET'])
def ingest_check():
    """Check if documents exist in Pinecone"""
    status = check_documents_exist()
    status['is_ingesting'] = _is_ingesting
    status['ingestion_complete'] = _ingestion_complete
    status['ingestion_error'] = _ingestion_error
    return jsonify(status)

@app.route('/api/ingest-run', methods=['POST'])
def ingest_run():
    """Manually trigger ingestion"""
    global _is_ingesting
    
    if _is_ingesting:
        return jsonify({
            'status': 'already_running',
            'message': 'Ingestion is already in progress'
        })
    
    run_ingestion()
    
    return jsonify({
        'status': 'started',
        'message': 'Ingestion started in background. Check /api/ingest-check for status'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" field'}), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Check if clients are available
        if _pinecone_index is None:
            return jsonify({
                'response': "⚠️ Pinecone is not available. Please check your configuration."
            })
        
        if _groq_client is None:
            return jsonify({
                'response': "⚠️ Groq is not available. Please check your configuration."
            })
        
        # Check if documents exist
        doc_status = check_documents_exist()
        
        if not doc_status['has_documents']:
            # Check if ingestion is running
            if _is_ingesting:
                return jsonify({
                    'response': "🔄 Documents are currently being ingested. Please wait a moment and try again.",
                    'document_status': doc_status,
                    'is_ingesting': True
                })
            else:
                return jsonify({
                    'response': "⚠️ No documents have been ingested yet. Run `python3 ingest_all.py` locally or POST to /api/ingest-run",
                    'document_status': doc_status
                })
        
        # Query Pinecone
        try:
            matches = query_pinecone(user_message, top_k=5)
        except Exception as e:
            return jsonify({
                'response': f"Error querying Pinecone: {str(e)}"
            }), 500
        
        if not matches:
            return jsonify({
                'response': "I could not find any relevant information in the documents for your question.",
                'sources': []
            })
        
        # Build context
        context_parts = []
        sources = []
        
        for match in matches:
            if match.get('score', 0) > 0.5:
                text = match.get('metadata', {}).get('text', '')
                if text:
                    context_parts.append(text)
                    sources.append({
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'score': match.get('score', 0)
                    })
        
        if not context_parts:
            return jsonify({
                'response': "I found some potentially relevant information, but none with high enough confidence. Please rephrase your question.",
                'sources': []
            })
        
        context = "\n\n---\n\n".join(context_parts[:3])
        
        # Generate response
        try:
            response = generate_response(user_message, context)
        except Exception as e:
            return jsonify({
                'response': f"Error generating response: {str(e)}"
            }), 500
        
        return jsonify({
            'response': response,
            'sources': sources,
            'document_status': doc_status,
            'sources_count': len(sources)
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ===== STARTUP =====
if __name__ == '__main__':
    # Check configuration
    missing_vars = []
    if not PINECONE_API_KEY:
        missing_vars.append('PINECONE_API_KEY')
    if not PINECONE_ENVIRONMENT:
        missing_vars.append('PINECONE_ENVIRONMENT')
    if not GROQ_API_KEY:
        missing_vars.append('GROQ_API_KEY')
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}", flush=True)
        print("Please check your .env file", flush=True)
    
    # Initialize clients
    initialize_clients()
    
    # Print status
    doc_status = check_documents_exist()
    print(f"📊 Document status: {doc_status['status']}", flush=True)
    if doc_status['has_documents']:
        print(f"   ✅ {doc_status['vector_count']} vectors loaded", flush=True)
    else:
        print("   ⚠️ No documents found", flush=True)
        if INGEST_ON_START:
            print("   🔄 Auto-ingestion started in background", flush=True)
        else:
            print("   ℹ️ Set INGEST_ON_START=true to auto-ingest", flush=True)
    
    print(f"🔍 Embedding model: {GROQ_EMBEDDING_MODEL}", flush=True)
    print(f"💬 Chat model: {GROQ_CHAT_MODEL}", flush=True)
    
    # Start the Flask app
    print("🚀 Starting Flask server on port 5000...", flush=True)
    app.run(debug=False, host='0.0.0.0', port=5000)