from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_JWT_SECRET: str
    SUPABASE_SERVICE_ROLE_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_HOST: str
    PINECONE_INDEX_NAME: str = "knowledge-brain"
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    PDF_DIRECTORY: str = "./pdfs"
    MAX_RETRIEVAL_DOCS: int = 10
    MAX_RELEVANT_DOCS: int = 5
    MAX_REVISION_RETRIES: int = 1
    MAX_REWRITE_RETRIES: int = 1

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()