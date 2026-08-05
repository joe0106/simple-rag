from typing import List
from fastapi import APIRouter, HTTPException
from components.query_remaster import QueryRemaster
from components.retriever import Retriever
from components.evaluator import Evaluator
from components.response_generator import LLMResponseGenerotor
from components.rag_workflow import RAGWorkFlow
from config import Settings
from models import (
    QueryRequest,
    RAGResponse,
    RAGResponseStatus,
    Document,
    Collection,
    DocumentPathLoadRequest,
)


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
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        response = workflow.execute(request.query)
    except (TimeoutError, ConnectionError) as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG dependency is temporarily unavailable",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="query processing failed",
        ) from exc

    if response is None:
        return RAGResponse(
            answer="I don't have enough information in the knowledge base to answer this question reliably.",
            citations=[],
            confidence_score=0.0,
            status=RAGResponseStatus.INSUFFICIENT_CONTEXT,
            message="The retrieved context did not meet the completion threshold.",
        )

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
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document file not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (TimeoutError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail="vector database is temporarily unavailable") from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail="document loading failed") from e

@docRouter.post("")
async def create_document(request: List[Document]):
    """Add a new document into chroma db \"documents\" collection"""
    if not request:
        raise HTTPException(status_code=400, detail="at least one document is required")

    try:
        ids = await retriever.add_documents(request)
        if not ids:
            raise HTTPException(status_code=500, detail="document IDs were not returned")
        return f"document: {ids} created successfully"
    except HTTPException:
        raise
    except (TimeoutError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail="vector database is temporarily unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="document add request failed") from exc

@docRouter.get("")
def get_collection_documents():
    return retriever.get_collection_documents()

@docRouter.delete("")
def reset_collection():
    """reset the whole vector db collection"""
    try:
        retriever.reset_collection()
        return "vector db reset"
    except (TimeoutError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail="vector database is temporarily unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="vector database reset failed") from exc

@docRouter.delete("/id/{id}")
async def delete_doc_by_id(id: str):
    """delete docs by vector IDs"""
    try:
        await retriever.delete_doc_by_ids(id)
        return f"document deleted, id: {id}"
    except (TimeoutError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail="vector database is temporarily unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="document deletion failed") from exc

# COLLECTION Router
@collectionRouter.get("")
def get_collections():
    """list collections in vector db"""
    try:
        collections = retriever.get_collections()
    except (TimeoutError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail="vector database is temporarily unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="collection listing failed") from exc

    return [
        Collection(
            id=str(collection.id),
            name=collection.name,
            database=getattr(collection, "database", "default_database"),
            metadata=collection.metadata or {},
        )
        for collection in collections
    ]
