from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LCD
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from models import SearchResult, Document

class Retriever():
    def __init__(
        self,
        api_key: str,
        embed_model: str = "gemini-embedding-001",
        vector_store: str = None,
        top_k: int = 2,
        candidate_k: int = 10,
        ensemble_weights: List[float] | None = None,
        cohere_api_key: str | None = None,
        reranker_model: str = "rerank-v3.5",
    ):
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")

        self._embeds = GoogleGenerativeAIEmbeddings(api_key=api_key, model=embed_model)
        self._cohere_reranker = CohereRerank(
            model=reranker_model,
            top_n=top_k,
            cohere_api_key=cohere_api_key,
        )
        self._chroma_db = Chroma(
            collection_name="documents",
            embedding_function=self._embeds,
            persist_directory=vector_store
        )
        self.persist_dir = vector_store
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.ensemble_weights = ensemble_weights or [0.5, 0.5]
        self._bm25_retriever = None
        self._vector_retriever = None
        self._ensemble_retriever = None
        self._refresh_retrievers()

    async def execute(self, query: str, keywords: List[str]):
        #semantic_search_result = await self.execute_semantic_search(query)
        #keyword_search_result = await self.execute_keyword_search(query, keywords)
        return await self.retrieve(query, keywords)

    async def execute_semantic_search(self, query: str):
        return await self.semantic_search(query)

    async def execute_keyword_search(self, query: str, keywords: List[str]):
        return await self.keyword_search(query, keywords)

    async def retrieve(self, query: str, keywords: List[str]):
        """Retrieve context through the hybrid search pipeline."""
        return await self.hybrid_search(query, keywords)

    async def semantic_search(self, query: str) -> List[SearchResult]:
        results = await self._vector_retriever.ainvoke(query)
        return self._documents_to_search_results(results)

    async def keyword_search(self, query: str, keywords: List[str]) -> List[SearchResult]:
        query = " ".join([query, *keywords]).strip()
        results = await self._bm25_retriever.ainvoke(query)
        return self._documents_to_search_results(results)

    async def hybrid_search(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Fuse semantic and keyword results, then rerank candidates with Cohere."""
        if self._ensemble_retriever is None:
            return []

        ensemble_query = " ".join([query, *keywords]).strip()
        documents = await self._ensemble_retriever.ainvoke(ensemble_query)
        if not documents:
            return []

        fallback_results = self._documents_to_search_results(documents)
        try:
            reranked_documents = await self._cohere_reranker.acompress_documents(
                documents,
                ensemble_query,
            )
        except Exception:
            return fallback_results[:self.top_k]

        return self._documents_to_search_results(reranked_documents)[:self.top_k]

    @staticmethod
    def _documents_to_search_results(documents: List[LCD]) -> List[SearchResult]:
        results = []
        for rank, document in enumerate(documents):
            score = document.metadata.get("relevance_score")
            if not isinstance(score, (int, float)):
                score = 1.0 / (rank + 1)
            results.append(SearchResult(
                text=document.page_content,
                metadata=document.metadata,
                score=float(score),
            ))
        return results

    def _refresh_retrievers(self):
        """Rebuild the sparse retriever after the Chroma collection changes."""
        vector_retriever = self._chroma_db.as_retriever(
            search_kwargs={"k": self.candidate_k}
        )
        collection = self._chroma_db.get(
            include=["documents", "metadatas"]
        )
        page_contents = collection.get("documents") or []
        metadatas = collection.get("metadatas") or []
        documents = [
            LCD(
                page_content=page_content,
                metadata=metadatas[index] or {} if index < len(metadatas) else {},
            )
            for index, page_content in enumerate(page_contents)
            if page_content is not None
        ]

        if documents:
            self._vector_retriever = vector_retriever
            self._bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=self.candidate_k,
            )
            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                weights=self.ensemble_weights,
            )
        else:
            self._bm25_retriever = None
            self._ensemble_retriever = vector_retriever

    def load_predefined_docs(self):
        docs = [LCD(
                    page_content="""
                    Hello, my name is Julian Vance, and I am an undergraduate student currently pursuing a Bachelor of Science in Economics with a minor in International Relations. Driven by a keen interest in global market trends and public policy, I have dedicated my academic career to mastering quantitative data analysis and economic modeling. My long-term professional objective is to contribute to macroeconomic research and strategic financial consulting within an international institution. I consistently look for opportunities to bridge the gap between rigorous academic theory and practical, real-world fiscal applications.
                    """,
                    metadata={"source": "student"}
                ),
                LCD(
                    page_content="""
                    Organization: The Undergraduate Economics & Finance SocietyCurrent Role: Vice President of Research & AnalyticsKey Duties: Managing a team of student analysts to publish quarterly campus market reports.Secondary Club: Model United Nations (Active Member, Finance Committee Lead)
                    """,
                    metadata={"source": "club"}
                ),
                LCD(
                    page_content="""
                    Institution: Prestige State UniversityAcademic Year: Junior (Third Year)Current GPA: 3.87 / 4.00Expected Graduation: May 2028
                    """,
                    metadata={"source": "university"}
                )]
        ids = self._chroma_db.add_documents(
            documents=docs
        )
        self._refresh_retrievers()
        return ids

    async def add_documents(self, docs: List[Document]) -> List[str]:
        # model transformation from custom to langchain_core.documents.Documents
        docs = [
            LCD(page_content=doc.text, metadata=doc.metadata)
            for doc in docs
        ]

        ids = await self._chroma_db.aadd_documents(
            documents=docs
        )
        self._refresh_retrievers()
        return ids

    async def load_pdf_by_local_path(self, file: str, file_ext: str, size: int = 500, overlap: int = 50):
        dir_path = Path(__file__).parent.parent.resolve()
        file_path = Path(f"{dir_path}/docs/{file}.{file_ext}")
        if not file_path.is_file():
            raise FileNotFoundError("file not found")
        if file_ext != "pdf":
            raise Exception(f"Cannot proccess requested file ext: {file_ext}")
        loader = PyPDFLoader(file_path)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = size,
            chunk_overlap = overlap
        )
        chunks = text_splitter.split_documents(doc)
        ids = await self._chroma_db.aadd_documents(
            documents=chunks
        )
        self._refresh_retrievers()
        return ids

    async def delete_doc_by_ids(self, doc_id: str):
        result = await self._chroma_db.adelete(doc_id)
        self._refresh_retrievers()
        return result

    def reset_collection(self):
        result = self._chroma_db.reset_collection()
        self._refresh_retrievers()
        return result

    def get_collection_documents(self):
        return self._chroma_db.get()

    def get_collections(self):
        return self._chroma_db._client.list_collections()
