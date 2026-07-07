from flask import Blueprint, request, jsonify
from auth import require_auth
from rag_engine import SelfRAGEngine
from memory import ConversationMemory
from db import supabase_admin
from vector_store import get_all_filenames
import re
import logging
import traceback

logger = logging.getLogger(__name__)
chat_bp = Blueprint('chat', __name__, url_prefix='/api')

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

    try:
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

        # Save conversation only if answer is not a failure message
        if result.retrieval_used and result.answer != "I don't have an answer related to this question.":
            engine.memory.add_message(session_id, "user", question)
            engine.memory.add_message(session_id, "assistant", result.answer)

        # Only send sources if we actually have a real answer
        sources = result.sources if result.answer != "I don't have an answer related to this question." else []

        return jsonify({
            "answer": result.answer,
            "sources": sources,
            "metadata": {
                "retrieval_used": result.retrieval_used,
                "learned": result.learned
            }
        })

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}\n{traceback.format_exc()}")
        return jsonify({"answer": "I don't have an answer related to this question.", "sources": []}), 200