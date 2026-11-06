"""Environment-driven settings. Secrets never live in code."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    request_timeout_s: float = 60.0
    max_retries: int = 3
    max_message_chars: int = 2000
    rate_limit_per_min: int = 20
    max_sessions: int = 1000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
