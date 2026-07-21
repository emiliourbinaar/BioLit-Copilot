from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIOLIT_", env_file=".env", extra="ignore")

    ncbi_api_key: str | None = None
    ncbi_tool: str = "biolit-copilot"
    ncbi_email: str | None = None

    http_max_retries: int = 4
    http_backoff_base_seconds: float = 0.5
    http_backoff_max_seconds: float = 8.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
