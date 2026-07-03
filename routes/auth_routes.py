from flask import Blueprint, request, jsonify, make_response
from auth import require_auth
from db import supabase, supabase_admin
import uuid

auth_bp = Blueprint('auth', __name__, url_prefix='/api')

@auth_bp.route('/signup', methods=['POST', 'OPTIONS'])
def signup():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    response = supabase.auth.sign_up({"email": email, "password": password})
    if response.user:
        api_key = str(uuid.uuid4()).replace('-', '')[:32]
        supabase_admin.table('users').insert({
            'id': response.user.id,
            'email': email,
            'api_key': api_key
        }).execute()
        return jsonify({"user": {"email": response.user.email, "id": response.user.id}})
    return jsonify({"error": "Sign-up failed"}), 400

@auth_bp.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
    if response.user:
        api_key = str(uuid.uuid4()).replace('-', '')[:32]
        supabase_admin.table('users').upsert({
            'id': response.user.id,
            'email': email,
            'api_key': api_key
        }).execute()
        resp = make_response(jsonify({
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "user": {"email": response.user.email, "id": response.user.id}
        }))
        resp.set_cookie('chatbot_token', response.session.access_token,
                        httponly=True, secure=False, samesite='Lax', max_age=60*60*24*7)
        return resp
    return jsonify({"error": "Invalid credentials"}), 401

@auth_bp.route('/logout', methods=['POST'])
@require_auth
def logout():
    supabase.auth.sign_out()
    resp = jsonify({"success": True})
    resp.set_cookie('chatbot_token', '', expires=0)
    return resp

@auth_bp.route('/me', methods=['GET'])
@require_auth
def get_user():
    return jsonify({"user": {"email": request.user.email, "id": request.user.id}})

@auth_bp.route('/api_key', methods=['GET'])
@require_auth
def get_api_key():
    result = supabase_admin.table('users').select('api_key').eq('id', request.user.id).execute()
    if result.data:
        return jsonify({"api_key": result.data[0]['api_key']})
    return jsonify({"error": "No API key found"}), 404