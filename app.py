import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import pinecone
from groq import Groq
from typing import List, Dict, Any, Optional
import time
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ===== CONFIGURATION =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")

# Groq API Configuration (for BOTH embeddings and chat)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small")  # Groq embedding model
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "mixtral-8x7b-32768")  # Groq chat model

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Pinecone connection (lazy initialization)
_pinecone_index = None

# ===== PINEOCNE FUNCTIONS =====

def get_pinecone_index():
    """Lazy initialize Pinecone index"""
    global _pinecone_index
    if _pinecone_index is None:
        try:
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )
            _pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
            logger.info(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Pinecone: {e}")
            raise
    return _pinecone_index

def check_documents_exist() -> Dict[str, Any]:
    """Check if documents exist in Pinecone (serverless check only)"""
    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
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
            'message': f'Error checking documents: {str(e)}'
        }

# ===== GROQ EMBEDDINGS =====

@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    """Get embedding using Groq API"""
    try:
        # Groq embeddings endpoint
        response = groq_client.embeddings.create(
            model=GROQ_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding from Groq: {e}")
        raise

def query_pinecone(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query Pinecone for relevant documents"""
    try:
        index = get_pinecone_index()
        query_embedding = get_embedding(query)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.get('matches', [])
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return []

# ===== GROQ CHAT =====

def generate_response(query: str, context: str) -> str:
    """Generate a response using Groq API with context"""
    try:
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
        If the context doesn't contain the answer, say "I could not find that information in the documents."
        Be concise and accurate. Use the context to support your answers."""
        
        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer based only on the context provided."""
        
        # Using Groq for chat
        response = groq_client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

# ===== API ROUTES =====

@app.route('/')
def home():
    """Home page - serves the chat interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pinecone_configured': all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]),
        'groq_configured': bool(GROQ_API_KEY),
        'groq_embedding_model': GROQ_EMBEDDING_MODEL,
        'groq_chat_model': GROQ_CHAT_MODEL
    })

@app.route('/api/ingest-check', methods=['GET'])
def ingest_check():
    """Check if documents exist in Pinecone (serverless check only)"""
    try:
        status = check_documents_exist()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'has_documents': False,
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

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
        
        # First check if documents exist
        doc_status = check_documents_exist()
        
        if not doc_status['has_documents']:
            return jsonify({
                'response': "⚠️ No documents have been ingested yet. Please run `python3 ingest_all.py` locally first.",
                'document_status': doc_status
            })
        
        # Query Pinecone for relevant context using Groq embeddings
        matches = query_pinecone(user_message, top_k=5)
        
        if not matches:
            return jsonify({
                'response': "I could not find any relevant information in the documents for your question.",
                'sources': []
            })
        
        # Build context from matches
        context_parts = []
        sources = []
        
        for match in matches:
            if match.get('score', 0) > 0.5:  # Only use high-confidence matches
                text = match.get('metadata', {}).get('text', '')
                if text:
                    context_parts.append(text)
                    sources.append({
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'score': match.get('score', 0)
                    })
        
        if not context_parts:
            return jsonify({
                'response': "I found some potentially relevant information, but none with high enough confidence to use. Please rephrase your question.",
                'sources': []
            })
        
        # Combine context
        context = "\n\n---\n\n".join(context_parts[:3])  # Limit to top 3 chunks
        
        # Generate response using Groq
        response = generate_response(user_message, context)
        
        return jsonify({
            'response': response,
            'sources': sources,
            'document_status': doc_status,
            'models_used': {
                'embedding': GROQ_EMBEDDING_MODEL,
                'chat': GROQ_CHAT_MODEL
            }
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get detailed status including document count"""
    try:
        doc_status = check_documents_exist()
        
        return jsonify({
            'pinecone_connected': True,
            'document_status': doc_status,
            'config': {
                'pinecone_environment': PINECONE_ENVIRONMENT,
                'pinecone_index': PINECONE_INDEX_NAME,
                'groq_configured': bool(GROQ_API_KEY),
                'groq_embedding_model': GROQ_EMBEDDING_MODEL,
                'groq_chat_model': GROQ_CHAT_MODEL
            }
        })
    except Exception as e:
        return jsonify({
            'pinecone_connected': False,
            'error': str(e)
        }), 500

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ===== FOR LOCAL DEVELOPMENT =====

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
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
    
    # Check if documents exist
    doc_status = check_documents_exist()
    if doc_status['has_documents']:
        print(f"✅ Documents loaded: {doc_status['vector_count']} vectors")
    else:
        print("⚠️ No documents found. Run: python3 ingest_all.py")
    
    # Print Groq model info
    print(f"🔍 Embedding model: {GROQ_EMBEDDING_MODEL}")
    print(f"💬 Chat model: {GROQ_CHAT_MODEL}")
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)