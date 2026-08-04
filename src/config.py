from pathlib import Path
from pydantic_settings import BaseSettings

env_file_name = ".env"

class Settings(BaseSettings):
    google_api_key: str
    vector_store: str

    #Component model
    remaster_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    evaluator_model: str = "gemini-3.5-flash"
    answer_model: str = "gemini-3.6-flash"

    #document processing
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = env_file_name