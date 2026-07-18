"""
Session REST router backed by Google Cloud Firestore database.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.schemas import CEFRLevel, ChatMessage, RolePlayMode
from app.services.firebase import (
    get_current_user_id,
    save_user_session,
    get_user_session,
    get_user_dashboard
)

router = APIRouter(prefix="/session", tags=["session"])


class Session(BaseModel):
    session_id: str
    user_id: str | None = None
    role_play_mode: RolePlayMode = RolePlayMode.free_talk
    cefr_level: CEFRLevel = CEFRLevel.B1
    history: list[ChatMessage] = Field(default_factory=list)
    word_bank: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


@router.get("/{session_id}")
async def get_session(session_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    """Fetch session history and states from Firestore."""
    sess = get_user_session(user_id, session_id)
    if sess:
        return sess
        
    # Return a default empty session structure if not found
    return {
        "session_id": session_id,
        "user_id": user_id,
        "role_play_mode": RolePlayMode.free_talk.value,
        "cefr_level": CEFRLevel.B1.value,
        "history": [],
        "word_bank": [],
        "metrics": {
            "total_turns": 0,
            "average_pronunciation_score": 0.0,
            "grammar_accuracy_rate": 0.0
        }
    }


@router.post("")
async def create_session(session: Session, user_id: str = Depends(get_current_user_id)) -> dict:
    """Save or update session payload in Firestore database."""
    session_data = session.model_dump()
    # Enforce token user ID match
    session_data["user_id"] = user_id
    save_user_session(user_id, session.session_id, session_data)
    return {
        "session_id": session.session_id,
        "stored": True
    }


@router.get("/{session_id}/dashboard")
async def get_dashboard(session_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    """Return cumulative learner metrics aggregated across all sessions from Firestore."""
    dash = get_user_dashboard(user_id)
    if dash:
        # Include session_id for contract completeness
        dash["session_id"] = session_id
        return dash
        
    return {
        "session_id": session_id,
        "total_turns": 0,
        "average_pronunciation_score": 0.0,
        "grammar_accuracy_rate": 0.0,
        "cefr_level": CEFRLevel.B1.value,
        "vocabulary_words_learned": 0
    }
