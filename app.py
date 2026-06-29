"""
RAG Chatbot with Groq API, Pinecone, and Auto-Ingestion
Deployable on Vercel - FIXED VERSION
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

# ===== GLOBAL STATE =====
_pinecone_index = None
_groq_client = None

# ===== INITIALIZE CLIENTS =====
def initialize_clients():
    """Initialize Groq and Pinecone clients"""
    global _pinecone_index, _groq_client
    
    try:
        # Groq client
        if GROQ_API_KEY:
            _groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("✅ Groq client initialized")
        else:
            logger.error("❌ GROQ_API_KEY not set")
        
        # Pinecone with new v5 syntax
        if PINECONE_API_KEY and PINECONE_INDEX_HOST:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            _pinecone_index = pc.Index(host=PINECONE_INDEX_HOST)
            logger.info(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' connected")
            
            # Check if documents exist
            stats = _pinecone_index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            logger.info(f"   📊 Vector count: {vector_count}")
        else:
            logger.error("❌ PINECONE_API_KEY or PINECONE_INDEX_HOST not set")
            
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        _pinecone_index = None
        _groq_client = None

# ===== GROQ FUNCTIONS =====
@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    """Get embedding using Pinecone's inference API (384-dim)"""
    global _pinecone_index
    
    if _pinecone_index is None:
        raise Exception("Pinecone index not available")
    
    try:
        # Use Pinecone's inference API for embeddings
        import requests
        url = f"https://{PINECONE_INDEX_HOST.split('https://')[-1]}/embeddings"
        headers = {
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "multilingual-e5-large",
            "parameters": {"input_type": "passage"},
            "inputs": [text[:8000]]
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["values"]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def query_pinecone(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query Pinecone for relevant documents"""
    global _pinecone_index
    
    if _pinecone_index is None:
        raise Exception("Pinecone index not available")
    
    # Get query embedding
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
        'vector_count': doc_status['vector_count']
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
        'vector_count': doc_status['vector_count']
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
                'error': 'Pinecone not connected'
            })
        
        stats = _pinecone_index.describe_index_stats()
        return jsonify({
            'total_documents': 10,  # From your metadata
            'total_chunks': stats.get('total_vector_count', 0),
            'total_pages': 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
                'sources': []
            })
        
        if _groq_client is None:
            return jsonify({
                'answer': "⚠️ Groq is not available. Please check your configuration.",
                'sources': []
            })
        
        # Check if documents exist
        doc_status = check_documents_exist()
        
        if not doc_status['has_documents']:
            return jsonify({
                'answer': "⚠️ No documents have been ingested yet. Please run ingestion first.",
                'sources': []
            })
        
        # Query Pinecone
        try:
            matches = query_pinecone(user_message, top_k=5)
        except Exception as e:
            return jsonify({
                'answer': f"Error querying Pinecone: {str(e)}",
                'sources': []
            }), 500
        
        if not matches:
            return jsonify({
                'answer': "I could not find any relevant information in the documents for your question.",
                'sources': []
            })
        
        # Build context
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
        
        # Generate response
        try:
            response = generate_response(user_message, context)
        except Exception as e:
            return jsonify({
                'answer': f"Error generating response: {str(e)}",
                'sources': []
            }), 500
        
        return jsonify({
            'answer': response,
            'sources': sources[:3],
            'sources_count': len(sources)
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

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
    
    print(f"💬 Chat model: {GROQ_CHAT_MODEL}", flush=True)
    
    # Start the Flask app
    print("🚀 Starting Flask server on port 5000...", flush=True)
    app.run(debug=False, host='0.0.0.0', port=5000)