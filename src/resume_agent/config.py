from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Prefer GOOGLE_API_KEY; GEMINI_API_KEY is also accepted by the Google SDK
    google_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    # minimal | low | medium | high — Flash-Lite defaults to minimal
    gemini_thinking_level: str = "minimal"

    @property
    def api_key(self) -> str:
        return self.google_api_key or self.gemini_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
