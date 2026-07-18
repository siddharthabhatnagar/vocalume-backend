"""Health & info endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.config import settings

router = APIRouter(tags=["meta"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "vocalume-backend", "version": __version__}


@router.get("/info")
async def info() -> dict:
    return {
        "service": "vocalume-backend",
        "version": __version__,
        "nim_configured": settings.nim_configured,
        "nim_model": settings.nim_llm_model,
        "riva_asr_available": bool(settings.riva_asr_url),
        "riva_tts_available": bool(settings.riva_tts_url),
        "max_history_turns": settings.max_history_turns,
    }