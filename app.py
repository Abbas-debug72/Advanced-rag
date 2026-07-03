from flask import Flask, redirect, url_for, render_template, request, jsonify
from flask_cors import CORS
from config import get_settings
from db import supabase
import os

# Import blueprints from routes package
from routes import auth_bp, chat_bp, widget_bp, dashboard_bp

def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key, Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(widget_bp)
    app.register_blueprint(dashboard_bp)

    # Simple routes
    @app.route('/')
    def index():
        token = request.cookies.get('chatbot_token')
        if token:
            try:
                user = supabase.auth.get_user(token)
                if user and user.user:
                    return redirect('/dashboard')
            except:
                pass
        return redirect('/login')

    @app.route('/login')
    def login_page():
        return render_template("login.html")

    @app.route('/signup')
    def signup_page():
        return render_template("signup.html")

    # Debug/status endpoints (can be moved to a separate blueprint)
    @app.route('/api/stats')
    @require_auth
    def stats():
        from vector_store import pinecone_index, documents_metadata
        s = pinecone_index.describe_index_stats()
        return jsonify({
            "total_documents": len(documents_metadata),
            "total_chunks": s.get('total_vector_count', 0)
        })

    @app.route('/api/documents')
    @require_auth
    def documents():
        from vector_store import documents_metadata
        docs = [{"filename": f, **meta} for f, meta in documents_metadata.items()]
        return jsonify({"documents": docs, "total": len(docs)})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=5000)