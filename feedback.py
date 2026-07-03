from db import supabase_admin
from datetime import datetime
from typing import Optional

def record_feedback(user_id: str, session_id: str, rating: int, corrected_answer: Optional[str] = None):
    data = {
        'user_id': user_id,
        'session_id': session_id,
        'rating': rating,
        'corrected_answer': corrected_answer,
        'created_at': datetime.now().isoformat()
    }
    supabase_admin.table('feedback').insert(data).execute()

# Simple learning: after collecting feedback, you can reweight documents
# For example, if a document appears often in positive feedback, boost its score.
# This is a placeholder – you can implement a background job.
def update_weights_from_feedback():
    # Not implemented in full – but the structure is here.
    pass