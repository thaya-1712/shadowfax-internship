"""
Central configuration for the RAG assistant.
All tunables live here so the rest of the codebase never hardcodes magic values.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- LLM provider ---
    llm_provider: str = Field(default="anthropic", description="'anthropic' or 'openai'")
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-4-6")
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")

    # --- Embeddings ---
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # --- Chunking ---
    chunk_size: int = Field(default=800)          # characters
    chunk_overlap: int = Field(default=120)        # characters

    # --- Retrieval ---
    top_k_initial: int = Field(default=12)         # candidates pulled from FAISS
    top_k_final: int = Field(default=4)            # after reranking, sent to the LLM
    min_similarity_score: float = Field(default=0.15)

    # --- Storage ---
    upload_dir: str = Field(default="data/uploads")
    index_dir: str = Field(default="data/index")

    # --- App ---
    app_name: str = Field(default="ShadowFox RAG Assistant")
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
