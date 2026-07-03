from functools import wraps
from flask import request, jsonify
import jwt
from db import supabase, supabase_admin
from config import get_settings

settings = get_settings()

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
            except Exception:
                pass

        api_key = get_api_key_from_request()
        if api_key:
            try:
                result = supabase_admin.table('users').select('*').eq('api_key', api_key).execute()
                if result.data:
                    user_data = result.data[0]
                    # Create a simple user object
                    request.user = type('User', (), {
                        'id': user_data['id'],
                        'email': user_data['email'],
                    })()
                    return f(*args, **kwargs)
            except Exception:
                pass

        return jsonify({"error": "Missing or invalid authentication"}), 401
    return decorated