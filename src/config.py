from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
env_file_name = f"{PROJECT_ROOT}/.env"

class Settings(BaseSettings):
    google_api_key: str
    cohere_api_key: str
    vector_store: str

    #Component model
    remaster_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    evaluator_model: str = "gemini-3.5-flash-lite"
    reranker_model: str = "rerank-v3.5"
    answer_model: str = "gemini-3.5-flash-lite"

    #document processing
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = env_file_name
