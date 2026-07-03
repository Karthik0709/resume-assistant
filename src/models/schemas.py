from pydantic import BaseModel
from typing import Optional, List


class Document(BaseModel):
    """Represents a chunk from the resume"""
    id: str
    content: str          
    source: str           
    page_number: int      
    chunk_index: int     


class ChatRequest(BaseModel):
    """User's chat request"""
    query: str  
    chat_history: List[dict] = []  # Previous messages


class ChatResponse(BaseModel):
    """API response to user"""
    message: str
    references: List[Document] = []  # Which resume chunks were used