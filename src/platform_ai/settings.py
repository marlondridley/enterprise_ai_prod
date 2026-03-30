from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    APP_NAME: str = 'Enterprise AI Demo'
    APP_ENV: str = 'local'
    APP_HOST: str = '127.0.0.1'
    APP_PORT: int = 8000

    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_BASE_URL: str
    AZURE_OPENAI_CHAT_MODEL: str = 'gpt-4o-mini'
    AZURE_OPENAI_JUDGE_MODEL: str = 'gpt-4o-mini'
    AZURE_OPENAI_EXTRACTION_MODEL: str = 'gpt-4o-mini'

    AZURE_SEARCH_ENDPOINT: str | None = None
    AZURE_SEARCH_API_KEY: str | None = None
    AZURE_SEARCH_INDEX: str | None = None

    COSMOS_ENDPOINT: str | None = None
    COSMOS_KEY: str | None = None
    COSMOS_DATABASE: str | None = None
    COSMOS_CONTAINER: str | None = None

    APPLICATIONINSIGHTS_CONNECTION_STRING: str | None = None
    KEY_VAULT_URI: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
