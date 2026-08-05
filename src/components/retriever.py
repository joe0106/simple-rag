from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LCD
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from models import SearchResult, Document

class Retriever():
    def __init__(self, embed_model: str = "gemini-embedding-001", vector_store: str = None):
        self._embeds = GoogleGenerativeAIEmbeddings(model=embed_model)
        self._chroma_db = Chroma(
            collection_name="documents",
            embedding_function=self._embeds,
            persist_directory=vector_store
        )
        self.persist_dir = vector_store
        
    def execute(self, query: str, keywords: List[str]):
        self.retrieve(query, keywords)

    def retrieve(self, query: str, keywords: List[str]):
        semantic_res = self.semantic_search(query)
        keyword_res = self.keyword_search(keywords)
        reranked_res = self.rerank(semantic_res, keyword_res)
        return reranked_res
        
    def semantic_search(self, query: str, top_k: int = 2) -> List[SearchResult]:
        """similarity_search_with_score()預設使用的是歐式距離，回傳的分數是「距離」，越小代表越相關"""
        results = self._chroma_db.similarity_search_with_score(
            query=query,
            k=top_k
        )
        return [
            SearchResult(
                text = res[0].page_content,
                metadata = res[0].metadata,
                score = res[1]
            ) for res in results
        ]

    def keyword_search(self, keywords: str, top_k: int = 2) -> List[SearchResult]:
        """關鍵字過濾+語意排序"""
        filter = {
            "$or": [{"$contains": keyword} for keyword in keywords]
        }
        results = self._chroma_db.similarity_search(
            query=" ".join(keywords),
            k=top_k,
            where_document=filter
        )
        return [
            SearchResult(
                text = res.page_content,
                metadata = res.metadata,
                score = 0.9 # -> 不該直接與semantic的距離分數比較
            ) for res in results
        ]

    def rerank(self, semantic_results: List[SearchResult], keyword_results: List[SearchResult]) -> List[SearchResult]:
        merged = {}
        for res in semantic_results:
            merged[res.text] = res
        for res in keyword_results:
            if res.text not in merged or res.score < merged[res.text].score:
                merged[res.text] = res
        return sorted(merged.values(), key=lambda x: x.score)

    def hybrid_search(self):
        ...

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
        return self._chroma_db.add_documents(
            documents=docs
        )

    async def add_documents(self, docs: List[Document]) -> List[str]:
        # model transformation from custom to langchain_core.documents.Documents
        docs = [
            LCD(page_content=doc.text,)
            for doc in docs
        ]

        return await self._chroma_db.aadd_documents(
            documents=docs
        )

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
        return await self._chroma_db.aadd_documents(
            documents=chunks
        )

    async def delete_doc_by_ids(self, doc_id: str):
        return await self._chroma_db.adelete(doc_id)

    def reset_collection(self):
        self._chroma_db.reset_collection()

    def get_collection_documents(self):
        return self._chroma_db.get()

    def get_collections(self):
        return self._chroma_db._client.list_collections()
