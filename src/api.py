from typing import List
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from components.query_remaster import QueryRemaster
from components.retriever import Retriever
from components.evaluator import Evaluator
from components.response_generator import LLMResponseGenerator
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
remaster = QueryRemaster(settings.google_api_key, model=settings.remaster_model)
retriever = Retriever(settings.google_api_key, embed_model=settings.embedding_model, vector_store=settings.vector_store)
evaluator = Evaluator(settings.google_api_key, model=settings.evaluator_model)
llm_response_generator = LLMResponseGenerator(settings.google_api_key, model=settings.answer_model)

#init wf
workflow = RAGWorkFlow(
    remaster,
    retriever,
    evaluator,
    llm_response_generator
)

# PROMPT Router
@promptRouter.post("/query-stream")
async def process_query_stream(request: QueryRequest = Body(
    openapi_examples={
        "query-1": {
            "summary": "query-1",
            "value": {
                "query": "有關員工手冊請和我說明遠端辦公的規定"
            }
        }
    }
)) -> StreamingResponse:
    async def response_stream():
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query cannot be empty")
        try:
            async for chunk in workflow.execute_streaming(request.query):
                yield chunk
        except Exception:
            yield "\nStreaming LLM Response failed."

    # todo: proper response when 5xx?
    return StreamingResponse(
        response_stream(),
        media_type="text/plain"
    )

@promptRouter.post("/query", response_model=RAGResponse)
async def process_query(request: QueryRequest = Body(
    openapi_examples={
        "query-1": {
            "summary": "query-1",
            "value": {
                "query": "有關員工手冊請和我說明遠端辦公的規定"
            }
        }
    }
)):
    """execute the RAG workflow by passing in the prompt to RAGWorkflow"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        response = await workflow.execute(request.query)
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

# DOCUMENT Router
@docRouter.post("/predefined")
async def load_predefined_documents():
    """load pre-defined documents(same documents of \"create_document\" endpoint's example) as the RAG knowledge base"""
    loaded_docs = retriever.load_predefined_docs()
    return f"{len(loaded_docs)} pre-defined documents loaded"

@docRouter.post("/load_doc")
async def load_documents_by_filepath(request: DocumentPathLoadRequest = Body(
    openapi_examples={
        "sample.doc": {
            "summary": "sample doc path",
            "value": {
                "path": "sample_doc",
                "extension": "pdf"
            }
        }
    }
)):
    """load a \"TechCorp Employee Handbook\" document from project path"""
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
async def create_document(request: List[Document] = Body(
    openapi_examples={
        "student": {
            "summary": "the student's document",
            "value": {
                "text": "Hello, my name is Julian Vance, and I am an undergraduate student currently pursuing a Bachelor of Science in Economics with a minor in International Relations. Driven by a keen interest in global market trends and public policy, I have dedicated my academic career to mastering quantitative data analysis and economic modeling. My long-term professional objective is to contribute to macroeconomic research and strategic financial consulting within an international institution. I consistently look for opportunities to bridge the gap between rigorous academic theory and practical, real-world fiscal applications.",
                "metadata": {
                    "source": "student"
                }
            }
        },
        "club": {
            "summary": "the student's club document",
            "value": {
                "text": "Organization: The Undergraduate Economics & Finance SocietyCurrent Role: Vice President of Research & AnalyticsKey Duties: Managing a team of student analysts to publish quarterly campus market reports.Secondary Club: Model United Nations (Active Member, Finance Committee Lead)",
                "metadata": {
                    "source": "club"
                }
            }
        },
        "university": {
            "summary": "the student's university document",
            "value": {
                "text": "Institution: Prestige State UniversityAcademic Year: Junior (Third Year)Current GPA: 3.87 / 4.00Expected Graduation: May 2028",
                "metadata": {
                    "source": "university"
                }
            }
        }
    }
)):
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
    """get current documents of the default \"docments\" collection"""
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
    """list all collections in the vector db"""
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
