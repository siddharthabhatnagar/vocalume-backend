"""
Session REST contract (placeholder).

Vercel serverless functions are stateless and short-lived -- there is
no local filesystem persistence across invocations (aside from
`/tmp`, which is ephemeral per-instance), so this router cannot
actually persist session history, word banks, or progress metrics. It
exists to define the REST *contract* the Android client can code
against today, with responses that are explicit about what is and
isn't persisted. In production, back this router with a real database
(e.g. Postgres via a managed provider, or Firebase Firestore, which
the VocaLume/Buildify stack already uses elsewhere) and replace the
placeholder logic with real reads/writes.

Until then, the Android client should keep the canonical copy of
session history, word bank, and metrics in its local Room database and
treat this endpoint purely as a future sync point.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas import CEFRLevel, ChatMessage, RolePlayMode

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
async def get_session(session_id: str) -> dict:
    """Placeholder session fetch -- no server-side persistence on Vercel."""
    return {
        "session_id": session_id,
        "persisted": False,
        "note": (
            "Vercel serverless functions cannot persist state between "
            "invocations. This endpoint is a contract placeholder -- store "
            "session history, word bank, and progress locally in the "
            "Android client's Room database, and wire a real database "
            "(e.g. Postgres) here for production sync."
        ),
    }


@router.post("")
async def create_session(session: Session) -> dict:
    """Placeholder session creation -- acknowledges receipt without persisting."""
    return {
        "session_id": session.session_id,
        "stored": False,
        "note": "Session payload received but not persisted (no database configured).",
    }


@router.get("/{session_id}/dashboard")
async def get_dashboard(session_id: str) -> dict:
    """Placeholder learner progress dashboard with a zeroed sample shape."""
    return {
        "session_id": session_id,
        "total_turns": 0,
        "average_pronunciation_score": 0.0,
        "grammar_accuracy_rate": 0.0,
        "cefr_level": CEFRLevel.B1.value,
        "vocabulary_words_learned": 0,
        "note": (
            "This is a zeroed placeholder shape. Wire a Postgres (or "
            "Firestore) backend to compute and persist real metrics across "
            "sessions in production."
        ),
    }
