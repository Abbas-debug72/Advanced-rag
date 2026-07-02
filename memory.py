# memory.py – Supabase-backed conversation memory
import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from supabase import Client

class ConversationMemory:
    """
    Manages conversation history stored in Supabase.
    Each user can have multiple sessions (identified by session_id).
    """
    
    def __init__(self, user_id: str, supabase_admin: Client, table_name: str = "conversations"):
        self.user_id = user_id
        self.supabase = supabase_admin
        self.table_name = table_name
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Insert a new message into the conversation."""
        try:
            data = {
                "user_id": self.user_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": datetime.now().isoformat()
            }
            self.supabase.table(self.table_name).insert(data).execute()
            return True
        except Exception as e:
            print(f"Error saving message: {e}")
            return False
    
    def get_history(self, session_id: str, last_n: int = 5) -> List[Dict]:
        """Retrieve the most recent messages for a session."""
        try:
            result = self.supabase.table(self.table_name) \
                .select("*") \
                .eq("user_id", self.user_id) \
                .eq("session_id", session_id) \
                .order("created_at", desc=True) \
                .limit(last_n) \
                .execute()
            # Return in chronological order (oldest first)
            messages = result.data
            messages.reverse()
            return messages
        except Exception as e:
            print(f"Error fetching history: {e}")
            return []
    
    def format_history(self, session_id: str, last_n: int = 5) -> str:
        """Format history as a string for the prompt."""
        history = self.get_history(session_id, last_n)
        if not history:
            return "No previous conversation."
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
    
    def clear_session(self, session_id: str) -> bool:
        """Delete all messages for a given session."""
        try:
            self.supabase.table(self.table_name) \
                .delete() \
                .eq("user_id", self.user_id) \
                .eq("session_id", session_id) \
                .execute()
            return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    def get_all_sessions(self) -> List[str]:
        """Return a list of distinct session IDs for this user."""
        try:
            result = self.supabase.table(self.table_name) \
                .select("session_id") \
                .eq("user_id", self.user_id) \
                .execute()
            sessions = set()
            for row in result.data:
                sessions.add(row["session_id"])
            return list(sessions)
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []