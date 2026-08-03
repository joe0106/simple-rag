from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from components.query_remaster import QueryRemaster
from components.retriever import Retriever
from components.evaluator import Evaluator
from components.response_generator import LLMResponseGenerotor
from components.rag_workflow import RAGWorkFlow
from models import Document
from typing import List
from config import Settings

app = FastAPI()

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

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    documents: List[Document]

@app.get("/hello-world")
def hello_world():
    return "hello from FastAPI"

@app.post("/query")
def process_query(request: QueryRequest):
    """execute the RAG workflow"""
    response = workflow.execute(request.query)
    if not response:
        return HTTPException(status_code=400, detail="query cannot be processed")
    return response

@app.post("/documents")
def load_documents():
    """load pre-defined documents as the RAG knowledge base"""
    retriever.addDocuments()
    return "Done"