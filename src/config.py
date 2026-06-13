from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    deepseek_api_key: str = Field(..., validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field("https://api.deepseek.com/v1", validation_alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field("deepseek-chat", validation_alias="DEEPSEEK_MODEL")
    chroma_persist_dir: str = Field(".chroma", validation_alias="CHROMA_PERSIST_DIR")
    knowledge_base_dir: str = Field("data/knowledge_base", validation_alias="KNOWLEDGE_BASE_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
