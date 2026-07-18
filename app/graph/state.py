"""
Shared conversation state for the LangGraph pipeline.

`ConversationState` is the TypedDict passed between the analyze,
classify, and respond nodes. It is intentionally `total=False` because
each node only needs to read a subset of fields and only returns the
subset it updates -- LangGraph merges partial node outputs into the
running state dict automatically. Fields map closely onto
`ChatRequest`/`ChatResponse` in `app/schemas.py`, plus a few pipeline-
internal fields (`confidence_map`) that never leave the graph.
"""

from typing import TypedDict

from app.schemas import CEFRLevel, FeedbackEvent, RolePlayMode


class ConversationState(TypedDict, total=False):
    session_id: str
    transcript: str
    history: list[dict]
    cefr_level: CEFRLevel
    role_play_mode: RolePlayMode
    user_profile: dict | None
    confidence_map: dict | None
    feedback: FeedbackEvent
    reply: str
