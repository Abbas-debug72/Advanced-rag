# memory.py
"""
Conversation memory manager for maintaining context
"""
import json
import os
from typing import List, Dict
from datetime import datetime
from collections import OrderedDict


class ConversationMemory:
    """
    Manages conversation history with:
    - Session-based memory
    - LRU eviction for memory management
    - Persistent storage
    """
    
    def __init__(self, storage_dir: str = "./conversations", max_sessions: int = 100):
        self.storage_dir = storage_dir
        self.max_sessions = max_sessions
        self.memory_file = os.path.join(storage_dir, "conversations.json")
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize or load conversations
        self.conversations = self._load()
    
    def _load(self) -> OrderedDict:
        """Load conversations from disk"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                return OrderedDict(data)
            except:
                pass
        return OrderedDict()
    
    def _save(self):
        """Save conversations to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump(dict(self.conversations), f, indent=2, default=str)
    
    def _evict_if_needed(self):
        """Remove oldest sessions if too many"""
        while len(self.conversations) > self.max_sessions:
            self.conversations.popitem(last=False)
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to a conversation session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            self._evict_if_needed()
        
        self.conversations[session_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        self.conversations[session_id]["last_updated"] = datetime.now().isoformat()
        self._save()
    
    def get_history(self, session_id: str, last_n: int = 5) -> List[Dict]:
        """Get recent conversation history"""
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id]["messages"]
        return messages[-last_n:]
    
    def format_history(self, session_id: str, last_n: int = 5) -> str:
        """Format conversation history for prompts"""
        history = self.get_history(session_id, last_n)
        
        if not history:
            return "No previous conversation."
        
        formatted = []
        for msg in history:
            if msg["role"] == "user":
                formatted.append(f"User: {msg['content']}")
            else:
                formatted.append(f"Assistant: {msg['content']}")
        
        return "\n".join(formatted)
    
    def clear_session(self, session_id: str):
        """Clear a specific conversation session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            self._save()
    
    def get_session_count(self) -> int:
        """Get total number of sessions"""
        return len(self.conversations)
    
    def get_total_messages(self) -> int:
        """Get total messages across all sessions"""
        return sum(
            len(session["messages"]) 
            for session in self.conversations.values()
        )