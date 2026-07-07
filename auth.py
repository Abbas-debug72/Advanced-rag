from functools import wraps
from flask import request, jsonify
from db import supabase, supabase_admin
import uuid

def get_token_from_request():
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ')[1]
    return request.cookies.get('chatbot_token')

def get_api_key_from_request():
    return (request.headers.get('X-API-Key') or
            request.headers.get('X-Api-Key') or
            request.headers.get('x-api-key'))

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        token = get_token_from_request()
        if token:
            try:
                user = supabase.auth.get_user(token)
                if user and user.user:
                    request.user = user.user
                    return f(*args, **kwargs)
            except Exception as e:
                print(f"JWT error: {e}")

        api_key = get_api_key_from_request()
        if not api_key:
            return jsonify({"error": "API key missing in request headers"}), 401

        try:
            result = supabase_admin.table('users').select('*').eq('api_key', api_key).execute()
            if result.data and len(result.data) > 0:
                user_data = result.data[0]
                request.user = type('User', (), {
                    'id': user_data['id'],
                    'email': user_data['email'],
                })()
                return f(*args, **kwargs)
            else:
                return jsonify({"error": "API key not found in database"}), 401
        except Exception as e:
            print(f"API key lookup error: {e}")
            return jsonify({"error": "Internal server error during key validation"}), 500

    return decorated