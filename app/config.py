"""Application settings, read from environment variables (or a .env file)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "TriagePilot"
    # SQLite by default so the app runs with zero setup; docker-compose switches to PostgreSQL.
    database_url: str = "sqlite:///./triagepilot.db"
    secret_key: str = "dev-only-secret-change-me-in-production-32b"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    model_path: str = "models/router.joblib"
    # Optional: when set, failure summaries are written by an LLM; otherwise a deterministic template is used.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-5-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
