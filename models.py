from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    metadata: dict = {}

class FeedbackRequest(BaseModel):
    session_id: str
    rating: int  # 1 for up, -1 for down
    corrected_answer: Optional[str] = None