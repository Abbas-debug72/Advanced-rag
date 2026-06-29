# app_server.py - Lightweight version for Vercel deployment
from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import json
from flask import Flask, request, jsonify, session
from memory import ConversationMemory
from pinecone import Pinecone
from groq import Groq

app = Flask(__name__)
app.secret_key = os.urandom(24)

print("🚀 Starting lightweight RAG server...")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-brain")

if not PINECONE_API_KEY or not PINECONE_INDEX_HOST:
    raise RuntimeError("Missing Pinecone configuration")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

# Initialize Groq
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not set")

groq_client = Groq(api_key=groq_api_key)
GROQ_MODEL = os.getenv("GROQ_CHAT_MODEL", "mixtral-8x7b-32768")
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small")

# Load document metadata (generated during local ingestion)
def load_document_metadata():
    metadata_file = "./brain_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

documents_metadata = load_document_metadata()
print(f"📚 Loaded metadata for {len(documents_metadata)} documents")

def get_all_filenames():
    return list(documents_metadata.keys())

def get_embedding(text: str):
    """Get embedding using Groq"""
    try:
        response = groq_client.embeddings.create(
            model=GROQ_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        raise

def search_pinecone(query: str, top_k: int = 5):
    """Search Pinecone"""
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

def generate_response(query: str, context: str):
    """Generate response using Groq"""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
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
        print(f"Generation error: {e}")
        raise

memory = ConversationMemory()

# Routes
@app.route("/")
def home():
    """Home endpoint"""
    return jsonify({
        "status": "ok",
        "service": "RAG Chatbot API",
        "documents_loaded": len(documents_metadata),
        "pinecone_connected": True,
        "groq_connected": True
    })

@app.route("/api/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "documents": len(documents_metadata)
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")

        if not question:
            return jsonify({"error": "Question required"}), 400

        # Search Pinecone
        matches = search_pinecone(question, top_k=5)
        
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
                page = match.get('metadata', {}).get('page_number', 1)
                if text:
                    context_parts.append(text)
                    sources.append({
                        'document': source_file,
                        'page': page,
                        'score': match.get('score', 0)
                    })

        if not context_parts:
            return jsonify({
                "answer": "I found some information but with low confidence. Please rephrase your question.",
                "sources": []
            })

        context = "\n\n---\n\n".join(context_parts[:3])
        answer = generate_response(question, context)

        # Add to memory
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", answer)

        return jsonify({
            "answer": answer,
            "sources": sources[:3],
            "sources_count": len(sources)
        })

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            "error": str(e),
            "answer": "An error occurred while processing your request."
        }), 500

@app.route("/api/stats", methods=["GET"])
def stats():
    """Get stats"""
    try:
        stats = index.describe_index_stats()
        return jsonify({
            "total_documents": len(documents_metadata),
            "total_chunks": stats.get('total_vector_count', 0),
            "documents": list(documents_metadata.keys())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/widget.js')
def serve_widget():
    """Serve the widget JavaScript file"""
    try:
        widget_path = os.path.join(os.path.dirname(__file__), 'widget.js')
        if os.path.exists(widget_path):
            with open(widget_path, 'r') as f:
                content = f.read()
            return content, 200, {'Content-Type': 'application/javascript'}
        return "Widget not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    print("\n🚀 Pinecone RAG Chatbot: http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)