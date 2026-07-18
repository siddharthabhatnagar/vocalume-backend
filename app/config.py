"""Centralized configuration loaded from environment variables.

Reads once at import time so serverless cold starts stay fast. All values have
sensible defaults that work on Vercel without any extra setup — except
``NVIDIA_API_KEY``, which is required for the LLM to do anything useful.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── NVIDIA NIM ──────────────────────────────────────────────────────
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nim_llm_model: str = Field(default="meta/llama-3.3-70b-instruct", alias="NIM_LLM_MODEL")
    nim_llm_temperature: float = Field(default=0.6, alias="NIM_LLM_TEMPERATURE")
    nim_llm_max_tokens: int = Field(default=512, alias="NIM_LLM_MAX_TOKENS")

    # ── Conversation limits ─────────────────────────────────────────────
    max_history_turns: int = Field(default=10, alias="MAX_HISTORY_TURNS")

    # ── CORS ────────────────────────────────────────────────────────────
    allowed_origins_raw: str = Field(default="*", alias="ALLOWED_ORIGINS")

    # ── Optional external Riva endpoints (Vercel cannot host Riva gRPC) ─
    riva_asr_url: str = Field(default="", alias="RIVA_ASR_URL")
    riva_tts_url: str = Field(default="", alias="RIVA_TTS_URL")

    @property
    def allowed_origins(self) -> List[str]:
        if self.allowed_origins_raw.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]

    @property
    def nim_configured(self) -> bool:
        return bool(self.nvidia_api_key and self.nvidia_api_key.startswith("nvapi-"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — instantiates once per cold start."""
    return Settings()


# Re-exported for convenience
settings = get_settings()