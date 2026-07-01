# brain.py – Simplified version (only for compatibility)
# This is a minimal version that just loads metadata
# The actual search is handled directly in app.py

import os
import json
from pathlib import Path
from typing import Dict, List

class KnowledgeBrain:
    """Simplified KnowledgeBrain that just loads metadata"""
    
    def __init__(
        self,
        pdf_directory: str = "./pdfs",
        embedding_model: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.pdf_directory = Path(pdf_directory)
        self.metadata_file = "./brain_metadata.json"
        self.documents_metadata = self._load_metadata()
        print(f"✅ Simplified Brain ready: {len(self.documents_metadata)} docs\n")
    
    def _load_metadata(self) -> dict:
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_stats(self) -> dict:
        return {
            "total_documents": len(self.documents_metadata),
            "total_chunks": 0,
            "total_pages": sum(m.get("pages", 0) for m in self.documents_metadata.values()),
            "categories": {}
        }
    
    def get_categories(self) -> List[str]:
        return sorted(set(m.get("category", "general") for m in self.documents_metadata.values()))
    
    def get_all_filenames(self) -> List[str]:
        return list(self.documents_metadata.keys())
    
    def search(self, query: str, k: int = 4, category: str = None) -> List:
        return []
    
    def intelligent_search(self, query: str, k: int = 4, category: str = None) -> List:
        return []