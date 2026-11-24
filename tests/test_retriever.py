import pytest

import components.retriever as retriever_module
from components.retriever import Retriever
from models import Document, SearchResult


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeEnsembleRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    async def ainvoke(self, query):
        self.queries.append(query)
        return self.documents


@pytest.mark.asyncio
async def test_hybrid_search_uses_ensemble_and_combines_keywords():
    retriever = object.__new__(Retriever)
    ensemble = FakeEnsembleRetriever([
        FakeDocument("semantic result", {"source": "vector"}),
        FakeDocument("keyword result", {"source": "bm25"}),
    ])
    retriever._ensemble_retriever = ensemble

    results = await retriever.hybrid_search("remote work", ["policy", "employee"])

    assert ensemble.queries == ["remote work policy employee"]
    assert [result.text for result in results] == ["semantic result", "keyword result"]
    assert [result.score for result in results] == [1.0, 0.5]
    assert results[0].metadata == {"source": "vector"}


@pytest.mark.asyncio
async def test_retrieve_delegates_to_hybrid_search(monkeypatch):
    retriever = object.__new__(Retriever)
    expected = [SearchResult(text="result", metadata={}, score=1.0)]

    async def fake_hybrid_search(query, keywords):
        assert query == "question"
        assert keywords == ["term"]
        return expected

    monkeypatch.setattr(retriever, "hybrid_search", fake_hybrid_search)

    assert await retriever.retrieve("question", ["term"]) == expected


@pytest.mark.asyncio
async def test_retrieve_returns_empty_for_empty_collection():
    retriever = object.__new__(Retriever)
    retriever._ensemble_retriever = None

    assert await retriever.retrieve("anything", []) == []


def test_refresh_retrievers_builds_bm25_from_collection(monkeypatch):
    retriever = object.__new__(Retriever)
    retriever.top_k = 2
    retriever.ensemble_weights = [0.5, 0.5]
    retriever._chroma_db = FakeChromaCollection([
        "employee handbook",
        "remote work policy",
    ], [{"source": "hr"}, {"source": "policy"}])

    monkeypatch.setattr(retriever_module, "BM25Retriever", FakeBM25Retriever)
    monkeypatch.setattr(retriever_module, "EnsembleRetriever", FakeEnsemble)

    retriever._refresh_retrievers()

    assert [doc.metadata for doc in retriever._bm25_retriever.documents] == [
        {"source": "hr"},
        {"source": "policy"},
    ]
    assert retriever._ensemble_retriever.weights == [0.5, 0.5]


def test_refresh_retrievers_handles_empty_collection():
    retriever = object.__new__(Retriever)
    retriever.top_k = 2
    retriever.ensemble_weights = [0.5, 0.5]
    retriever._chroma_db = FakeChromaCollection([], [])

    retriever._refresh_retrievers()

    assert retriever._bm25_retriever is None
    assert retriever._ensemble_retriever is retriever._chroma_db.vector_retriever


@pytest.mark.asyncio
async def test_add_documents_preserves_metadata_and_refreshes_index(monkeypatch):
    retriever = object.__new__(Retriever)
    retriever._chroma_db = FakeChroma()
    refreshed = []
    monkeypatch.setattr(retriever, "_refresh_retrievers", lambda: refreshed.append(True))

    ids = await retriever.add_documents([
        Document(text="handbook", metadata={"source": "hr"})
    ])

    assert ids == ["doc-1"]
    assert retriever._chroma_db.documents[0].metadata == {"source": "hr"}
    assert refreshed == [True]


class FakeChroma:
    def __init__(self):
        self.documents = []

    async def aadd_documents(self, documents):
        self.documents.extend(documents)
        return ["doc-1"]


class FakeChromaCollection:
    def __init__(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas
        self.vector_retriever = object()

    def as_retriever(self, search_kwargs):
        assert search_kwargs == {"k": 2}
        return self.vector_retriever

    def get(self, include):
        assert include == ["documents", "metadatas"]
        return {"documents": self.documents, "metadatas": self.metadatas}


class FakeBM25Retriever:
    def __init__(self, documents, k):
        self.documents = documents
        self.k = k

    @classmethod
    def from_documents(cls, documents, k):
        return cls(documents, k)


class FakeEnsemble:
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights
