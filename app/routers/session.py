"""Session management endpoints.

On Vercel serverless we cannot persist state in-process. These endpoints
provide a thin REST contract that the Android client uses with its own
backend (or a separate Postgres + Redis service) to persist conversation
history and progress metrics.

For the mini-project demo, these endpoints accept and return data without
persisting — the Android client is expected to store sessions locally in
Room DB. If you wire up a real database later, plug it in here.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/session", tags=["session"])


class Session(BaseModel):
    session_id: str
    user_id: str | None = None
    role_play_mode: str = "free_talk"
    cefr_level: str = "B1"
    history: list[dict[str, Any]] = Field(default_factory=list)
    word_bank: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "stored": False,
        "message": (
            "Vercel serverless cannot persist state in-process. The Android "
            "client should store sessions locally in Room DB. If you wire up "
            "Postgres, plug it in here."
        ),
    }


@router.post("")
async def save_session(session: Session) -> dict:
    return {"session_id": session.session_id, "stored": False, "acknowledged": True}


@router.get("/{session_id}/dashboard")
async def dashboard(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "total_sessions": 0,
        "avg_grammar_accuracy": 0.0,
        "avg_pronunciation_score": 0,
        "vocabulary_count": 0,
        "current_streak_days": 0,
        "cefr_history": [],
        "note": "Wire this endpoint to your Postgres/Redis store in production.",
    }