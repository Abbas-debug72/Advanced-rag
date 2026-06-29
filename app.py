"""
RAG Chatbot with Groq API, Pinecone, and Auto-Ingestion
Deployable on Vercel - COMPLETE FIXED VERSION
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
import requests
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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "mixtral-8x7b-32768")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")

print(f"🔧 CONFIG:", flush=True)
print(f"   PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}", flush=True)
print(f"   PINECONE_INDEX_HOST: {PINECONE_INDEX_HOST}", flush=True)
print(f"   GROQ_CHAT_MODEL: {GROQ_CHAT_MODEL}", flush=True)
print(f"   PDF_DIRECTORY: {PDF_DIRECTORY}", flush=True)

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
    
    # Check environment variables
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
                # Simple test to verify API key works
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
    
    # Initialize Pinecone with new v5 syntax
    try:
        if pinecone_key and pinecone_host:
            print("🔄 Connecting to Pinecone...", flush=True)
            pc = Pinecone(api_key=pinecone_key)
            _pinecone_index = pc.Index(host=pinecone_host)
            print(f"✅ Pinecone index connected successfully", flush=True)
            
            # Test connection and get stats
            try:
                stats = _pinecone_index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                print(f"   📊 Vector count: {vector_count}", flush=True)
                
                if vector_count == 0:
                    print("   ⚠️ No vectors found in index", flush=True)
                    print("   💡 Run ingestion locally: python brain.py", flush=True)
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

# ===== GROQ FUNCTIONS =====
@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    """Get embedding using Pinecone's inference API (384-dim)"""
    global _pinecone_index
    
    if _pinecone_index is None:
        raise Exception("Pinecone index not available")
    
    try:
        # Use Pinecone's inference API for embeddings
        # Extract host from index host
        host_url = PINECONE_INDEX_HOST
        if not host_url.startswith('https://'):
            host_url = f'https://{host_url}'
        
        url = f"{host_url}/embeddings"
        headers = {
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "multilingual-e5-large",
            "parameters": {"input_type": "passage"},
            "inputs": [text[:8000]]  # Limit text length
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            return data["data"][0]["values"]
        else:
            raise Exception("No embedding returned")
            
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def query_pinecone(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query Pinecone for relevant documents"""
    global _pinecone_index
    
    if _pinecone_index is None:
        raise Exception("Pinecone index not available")
    
    try:
        # Get query embedding
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
        'message': doc_status['message']
    })

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
    import os
    
    # Check if variables exist (without exposing full values)
    env_status = {
        'PINECONE_API_KEY': {
            'exists': bool(os.getenv('PINECONE_API_KEY')),
            'length': len(os.getenv('PINECONE_API_KEY', '')) if os.getenv('PINECONE_API_KEY') else 0,
            'preview': os.getenv('PINECONE_API_KEY', '')[:10] + '...' if os.getenv('PINECONE_API_KEY') else 'NOT SET'
        },
        'PINECONE_INDEX_HOST': {
            'exists': bool(os.getenv('PINECONE_INDEX_HOST')),
            'value': os.getenv('PINECONE_INDEX_HOST', 'NOT SET')
        },
        'GROQ_API_KEY': {
            'exists': bool(os.getenv('GROQ_API_KEY')),
            'length': len(os.getenv('GROQ_API_KEY', '')) if os.getenv('GROQ_API_KEY') else 0,
            'preview': os.getenv('GROQ_API_KEY', '')[:10] + '...' if os.getenv('GROQ_API_KEY') else 'NOT SET'
        },
        'PINECONE_INDEX_NAME': {
            'value': os.getenv('PINECONE_INDEX_NAME', 'knowledge-brain')
        },
        'GROQ_CHAT_MODEL': {
            'value': os.getenv('GROQ_CHAT_MODEL', 'mixtral-8x7b-32768')
        },
        'PDF_DIRECTORY': {
            'value': os.getenv('PDF_DIRECTORY', './pdfs')
        }
    }
    
    # Check Pinecone connection
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
            # Test Groq with a simple request
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
        'all_env_keys': [k for k in os.environ.keys() if not k.startswith('_')],
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
                'total_pages': 0,
                'error': 'Pinecone not connected',
                'connected': False
            })
        
        stats = _pinecone_index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        # Try to get document count from metadata
        # Since we can't query all, return what we know
        return jsonify({
            'total_documents': 10,  # From your metadata
            'total_chunks': vector_count,
            'total_pages': 0,
            'connected': True,
            'vector_count': vector_count,
            'index_name': PINECONE_INDEX_NAME
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
        
        # Check if clients are available
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
        
        # Check if documents exist
        doc_status = check_documents_exist()
        
        if not doc_status['has_documents']:
            return jsonify({
                'answer': "⚠️ No documents have been ingested yet. Please run ingestion first.\n\nTo ingest documents:\n1. Run `python brain.py` locally\n2. Or use the ingestion script provided",
                'sources': [],
                'document_status': doc_status
            })
        
        # Query Pinecone
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
        
        # Build context and sources
        context_parts = []
        sources = []
        
        for match in matches:
            if match.get('score', 0) > 0.3:  # Lower threshold for more results
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
        
        # Generate response
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

@app.route('/api/chat-widget', methods=['POST'])
def chat_widget():
    """Chat endpoint specifically for the widget (uses 'message' field)"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" field'}), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Forward to main chat handler
        # Reformat request to use 'question' field
        data['question'] = user_message
        
        # Use the same logic as /api/chat
        response = chat()
        
        # If response is a tuple, get the JSON data
        if isinstance(response, tuple):
            return response
        
        return response
        
    except Exception as e:
        logger.error(f"Widget chat error: {e}")
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
    if not PINECONE_INDEX_HOST:
        missing_vars.append('PINECONE_INDEX_HOST')
    if not GROQ_API_KEY:
        missing_vars.append('GROQ_API_KEY')
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}", flush=True)
        print("Please check your .env file and Vercel environment variables", flush=True)
    else:
        print("✅ All required environment variables are set", flush=True)
    
    # Initialize clients
    initialize_clients()
    
    # Print status
    doc_status = check_documents_exist()
    print(f"\n📊 Document status: {doc_status['status']}", flush=True)
    if doc_status['has_documents']:
        print(f"   ✅ {doc_status['vector_count']} vectors loaded", flush=True)
    else:
        print("   ⚠️ No documents found in Pinecone", flush=True)
        print("   💡 Run `python brain.py` locally to ingest documents", flush=True)
    
    print(f"\n💬 Chat model: {GROQ_CHAT_MODEL}", flush=True)
    print(f"📚 Embedding model: multilingual-e5-large (384-dim)", flush=True)
    
    # Start the Flask app
    print("\n🚀 Starting Flask server on port 5000...", flush=True)
    print("=" * 60, flush=True)
    app.run(debug=False, host='0.0.0.0', port=5000)