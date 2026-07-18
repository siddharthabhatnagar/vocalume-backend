"""LangGraph conversation state."""
from __future__ import annotations

from typing import Any, TypedDict

from app.schemas import CEFRLevel, FeedbackEvent, RolePlayMode


class ConversationState(TypedDict, total=False):
    session_id: str
    transcript: str
    history: list[dict[str, Any]]
    cefr_level: CEFRLevel
    role_play_mode: RolePlayMode
    user_profile: dict[str, Any] | None
    confidence_map: dict[str, float] | None
    feedback: FeedbackEvent
    reply: str