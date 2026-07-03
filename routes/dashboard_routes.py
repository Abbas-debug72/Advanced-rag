from flask import Blueprint, render_template, request
from auth import require_auth
from db import supabase_admin

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@require_auth
def dashboard():
    user_id = request.user.id
    api_key = None
    result = supabase_admin.table('users').select('api_key').eq('id', user_id).execute()
    if result.data:
        api_key = result.data[0]['api_key']
    return render_template("dashboard.html", user=request.user, api_key=api_key)