from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Document:
    text: str
    metadata: Dict = None

@dataclass
class SearchResult:
    text: str
    metadata: Dict
    score: float

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