"""Configuration via environment variables (AM_ prefix)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AM_", env_file=".env", extra="ignore")

    # sqlite = zero-config local backend; postgresql+psycopg://... for pgvector
    database_url: str = "sqlite:///./agent_memory.db"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536


@lru_cache
def get_settings() -> Settings:
    return Settings()
