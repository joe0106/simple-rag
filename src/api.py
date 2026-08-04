from typing import List
from fastapi import APIRouter, HTTPException
from components.query_remaster import QueryRemaster
from components.retriever import Retriever
from components.evaluator import Evaluator
from components.response_generator import LLMResponseGenerotor
from components.rag_workflow import RAGWorkFlow
from config import Settings
from models import QueryRequest, RAGResponse, Document, Collection, DocumentPathLoadRequest


promptRouter = APIRouter(
    prefix="/prompts",
    tags=["prompts"]
)
docRouter = APIRouter(
    prefix="/documents",
    tags=["documents"]
)
collectionRouter = APIRouter(
    prefix="/collections",
    tags=["collections"]
)
all_routers = [
    promptRouter, docRouter, collectionRouter
]

# load .env
settings = Settings()

#components
remaster = QueryRemaster(model=settings.remaster_model)
retriever = Retriever(embed_model=settings.embedding_model, vector_store=settings.vector_store)
evaluator = Evaluator(model=settings.evaluator_model)
llm_response_generator = LLMResponseGenerotor(model=settings.answer_model)

#init wf
workflow = RAGWorkFlow(
    remaster,
    retriever,
    evaluator,
    llm_response_generator
)

# PROMPT Router
@promptRouter.post("/query", response_model=RAGResponse)
async def process_query(request: QueryRequest):
    """execute the RAG workflow by passing in the prompt to RAGWorkflow"""
    response = workflow.execute(request.query)
    if not response:
        return HTTPException(status_code=400, detail="query cannot be processed")
    return response

# DOCUMEBNT Router
@docRouter.post("/predefined")
async def load_predefined_documents():
    """load pre-defined documents as the RAG knowledge base"""
    loaded_docs = retriever.load_predefined_docs()
    return f"{len(loaded_docs)} pre-defined documents loaded"

@docRouter.post("/load_doc")
async def load_documents_by_filepath(request: DocumentPathLoadRequest):
    """load document from project path"""
    try:
        doc_ids = await retriever.load_pdf_by_local_path(request.path.lower(), request.extension.lower())
        return f"{len(doc_ids)} doc/chunks loaded into vector database."
    except Exception as e:
        raise f"load_documents_by_filepath() failed: {e}"

@docRouter.post("")
async def create_document(request: List[Document]):
    """Add a new document into chroma db \"documents\" collection"""
    try:
        ids = await retriever.add_documents(request)
        if not ids:
            raise HTTPException(status_code=400, detail="failed to add document")
        return f"document: {ids} created successfully"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"document add request cannot be processed: {e}")

@docRouter.get("")
def get_collection_documents():
    return retriever.get_collection_documents()

@docRouter.delete("")
def reset_collection():
    """reset the whole vector db collection"""
    try:
        retriever.reset_collection()
        return "vector db reset"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"vector db cannot be deleted: {e}")

@docRouter.delete("/id/{id}")
async def delete_doc_by_id(id: str):
    """delete docs by vector IDs"""
    try:
        await retriever.delete_doc_by_ids(id)
        return f"document deleted, id: {id}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"documents cannot be deleted: {e}")

# COLLECTION Router
@collectionRouter.get("")
def get_collections():
    """list collections in vector db"""
    collections = retriever.get_collections()
    return [
        Collection(
            id=str(collection.id),
            name=collection.name,
            database=getattr(collection, "database", "default_database"),
            metadata=collection.metadata or {},
        )
        for collection in collections
    ]
