from enum import Enum
from typing import Any, Dict, List, Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass

class Document(BaseModel):
    text: str
    metadata: Annotated[Dict[str, Any], Field(default_factory=dict)]

@dataclass
class SearchResult:
    text: str
    metadata: Dict
    score: float

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    documents: List[Document]

class DocumentPathLoadRequest(BaseModel):
    path: str
    extension: str
    
class Citation(BaseModel):
    text: str
    metadata: Annotated[Dict[str, Any], Field(default_factory=dict)]
    relevance_score: float

class RAGResponseStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    NO_CONTEXT = "no_context"

class RAGResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_score: float = 0.0
    status: RAGResponseStatus = RAGResponseStatus.SUCCESS
    message: str = ""

class Collection(BaseModel):
    id: str
    name: str
    database: str
    metadata: Annotated[Dict[str, Any], Field(default_factory=dict)]
