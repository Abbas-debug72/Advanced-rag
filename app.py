from flask import Flask, redirect, url_for, render_template, request, jsonify
from flask_cors import CORS
from config import get_settings
from db import supabase
from auth import require_auth, get_api_key_from_request
import os
import traceback

# Import blueprints
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

    # --- Debug endpoint ---
    @app.route('/api/debug', methods=['GET'])
    def debug():
        try:
            from vector_store import pinecone_index, embedding_model, documents_metadata
            stats = pinecone_index.describe_index_stats()
            return jsonify({
                "pinecone_connected": True,
                "vector_count": stats.get('total_vector_count', 0),
                "documents": len(documents_metadata),
                "embedding_model_loaded": embedding_model is not None
            })
        except Exception as e:
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    @app.route('/api/test-auth', methods=['GET'])
    def test_auth():
        api_key = get_api_key_from_request()
        if not api_key:
            return jsonify({"error": "No API key sent"}), 400
        from db import supabase_admin
        try:
            result = supabase_admin.table('users').select('*').eq('api_key', api_key).execute()
            if result.data:
                return jsonify({"success": True, "user": result.data[0]['email']})
            else:
                return jsonify({"error": "Key not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Main routes ---
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

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=5000)