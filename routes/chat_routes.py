from flask import Blueprint, request, jsonify
from auth import require_auth
from rag_engine import SelfRAGEngine
from memory import ConversationMemory
from db import supabase_admin
from vector_store import get_all_filenames
import re

chat_bp = Blueprint('chat', __name__, url_prefix='/api')

# In-memory focus per session (can move to Redis later)
session_focus = {}

def detect_focus_command(question):
    q = question.lower()
    if "clear focus" in q:
        return "CLEAR"
    match = re.search(r'only\s+use\s+([\w\-.]+(?:\.pdf)?)', q)
    return match.group(1) if match else None

@chat_bp.route('/chat', methods=['POST', 'OPTIONS'])
@require_auth
def chat():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    data = request.get_json()
    question = data.get('question', '').strip()
    session_id = data.get('session_id', 'default')
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400

    focus_cmd = detect_focus_command(question)
    if focus_cmd == "CLEAR":
        session_focus.pop(session_id, None)
        return jsonify({"answer": "✅ Document filter cleared.", "sources": []})
    if focus_cmd:
        all_files = get_all_filenames()
        if focus_cmd in all_files:
            session_focus[session_id] = focus_cmd
            msg = f"✅ Now focusing on {focus_cmd}."
        else:
            msg = f"❌ Document '{focus_cmd}' not found."
        memory = ConversationMemory(request.user.id, supabase_admin)
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", msg)
        return jsonify({"answer": msg, "sources": []})

    focus_doc = session_focus.get(session_id)
    engine = SelfRAGEngine(request.user.id)
    result = engine.run(question, session_id, focus_doc)

    engine.memory.add_message(session_id, "user", question)
    engine.memory.add_message(session_id, "assistant", result.answer)

    return jsonify({
        "answer": result.answer,
        "sources": result.sources,
        "metadata": {
            "retrieval_used": result.retrieval_used,
            "support": result.support_verdict,
            "useful": result.useful,
            "retries": result.retries,
            "rewrite_tries": result.rewrite_tries
        }
    })

@chat_bp.route('/history', methods=['GET'])
@require_auth
def get_history():
    session_id = request.args.get('session_id', 'default')
    last_n = int(request.args.get('last_n', 10))
    memory = ConversationMemory(request.user.id, supabase_admin)
    history = memory.get_history(session_id, last_n)
    return jsonify({"history": history, "count": len(history)})

@chat_bp.route('/sessions', methods=['GET'])
@require_auth
def get_sessions():
    memory = ConversationMemory(request.user.id, supabase_admin)
    sessions = memory.get_all_sessions()
    return jsonify({"sessions": sessions})

@chat_bp.route('/conversation/<session_id>', methods=['DELETE'])
@require_auth
def clear_conversation(session_id):
    memory = ConversationMemory(request.user.id, supabase_admin)
    memory.clear_session(session_id)
    session_focus.pop(session_id, None)
    return jsonify({"success": True})

@chat_bp.route('/feedback', methods=['POST'])
@require_auth
def submit_feedback():
    data = request.get_json()
    session_id = data.get('session_id')
    rating = data.get('rating')
    corrected_answer = data.get('corrected_answer')
    if rating not in (1, -1):
        return jsonify({"error": "Rating must be 1 or -1"}), 400
    from feedback import record_feedback
    record_feedback(request.user.id, session_id, rating, corrected_answer)
    return jsonify({"success": True})