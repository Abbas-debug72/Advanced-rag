# learn_from_feedback.py
from db import supabase_admin
from collections import Counter

def analyze_feedback():
    # Get all feedback with negative rating
    result = supabase_admin.table('feedback').select('*').eq('rating', -1).execute()
    # Extract common patterns: which questions got negative feedback
    # You can also get the conversation history to see which documents were used.
    # This is a placeholder for your own logic.
    print(f"Found {len(result.data)} negative feedback entries.")
    # For each, you could mark those documents as less relevant (decrease weight)
    # Or you could use the corrected_answer to update the vector store with new chunks.

if __name__ == "__main__":
    analyze_feedback()