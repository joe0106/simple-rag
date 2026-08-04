from dataclasses import dataclass
from typing import Dict, List, Any
from pydantic import BaseModel

@dataclass
class Document:
    text: str
    metadata: Dict = None

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
    
@dataclass
class Citation:
    text: str
    metadata: Dict
    relevance_score: float

@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    confidence_score: float

class Collection(BaseModel):
    id: str
    name: str
    database: str
    metadata: Dict[str, Any]
