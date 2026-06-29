# memory.py – Vercel-Compatible (uses /tmp/)
import json
import os
from typing import List, Dict
from datetime import datetime
from collections import OrderedDict


class ConversationMemory:
    """Persistent conversation storage – Vercel compatible."""

    def __init__(self, storage_dir="/tmp/conversations", max_sessions=100):
        self.storage_dir = storage_dir
        self.max_sessions = max_sessions
        self.memory_file = os.path.join(storage_dir, "conversations.json")

        os.makedirs(storage_dir, exist_ok=True)
        self.conversations = self._load()

    def _load(self) -> OrderedDict:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return OrderedDict(data)
            except:
                pass
        return OrderedDict()

    def _save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.conversations), f, indent=2, default=str)

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
            }

        self.conversations[session_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        self.conversations[session_id]["last_updated"] = datetime.now().isoformat()

        while len(self.conversations) > self.max_sessions:
            self.conversations.popitem(last=False)

        self._save()

    def get_history(self, session_id: str, last_n: int = 5) -> List[Dict]:
        if session_id not in self.conversations:
            return []

        messages = self.conversations[session_id]["messages"]
        return messages[-last_n:]

    def format_history(self, session_id: str, last_n: int = 5) -> str:
        history = self.get_history(session_id, last_n)

        if not history:
            return "No previous conversation"

        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def clear_session(self, session_id: str):
        if session_id in self.conversations:
            del self.conversations[session_id]
            self._save()

    def get_session_count(self) -> int:
        return len(self.conversations)

    def get_total_messages(self) -> int:
        return sum(
            len(session["messages"])
            for session in self.conversations.values()
        )