from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Set OPENAI_ACTIVE=true with a valid OPENAI_API_KEY for live LLM; otherwise demo stubs are used.
    openai_active: bool = False
    openai_api_key: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
