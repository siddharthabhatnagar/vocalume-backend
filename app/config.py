"""
Application configuration for VocaLume backend.

Centralizes all environment-driven configuration using pydantic-settings
so that the rest of the codebase never touches `os.environ` directly.
Configuration covers the Cerebras Inference API connection (base URL,
model name, API key), per-use-case LLM sampling parameters (a low-
temperature "analyzer" profile for structured JSON tasks like grammar
correction, and a higher-temperature "conversational" profile for the
free-form role-play reply), conversation history truncation, and CORS
allowed origins. A single cached `Settings` instance is exposed via
`get_settings()` so importing modules share one object without re-
parsing the environment on every call.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed configuration for the VocaLume backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    cerebras_base_url: str = Field(
        default="https://inference.cerebras.ai/v1", alias="CEREBRAS_BASE_URL"
    )
    cerebras_model: str = Field(default="llama3.3-70b", alias="CEREBRAS_MODEL")

    llm_temperature_analyzer: float = Field(
        default=0.1, alias="LLM_TEMPERATURE_ANALYZER"
    )
    llm_temperature_conversation: float = Field(
        default=0.6, alias="LLM_TEMPERATURE_CONVERSATION"
    )
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")

    max_history_turns: int = Field(default=10, alias="MAX_HISTORY_TURNS")

    allowed_origins_raw: str = Field(default="*", alias="ALLOWED_ORIGINS")
    firebase_service_account_json: str = Field(default="", alias="FIREBASE_SERVICE_ACCOUNT_JSON")

    @property
    def allowed_origins(self) -> list[str]:
        """Return CORS allowed origins as a list, parsed from a comma-separated env var."""
        raw = (self.allowed_origins_raw or "*").strip()
        if raw == "*" or not raw:
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def cerebras_configured(self) -> bool:
        """True if a Cerebras API key of the expected shape has been configured."""
        return bool(self.cerebras_api_key) and self.cerebras_api_key.startswith("csk-")


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings singleton."""
    return Settings()
