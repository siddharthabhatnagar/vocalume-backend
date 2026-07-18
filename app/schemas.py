"""Pydantic request/response schemas for the VocaLume API.

These schemas are the wire contract between the Android (Jetpack Compose)
client and this FastAPI backend.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────
class CEFRLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class RolePlayMode(str, Enum):
    FREE_TALK = "free_talk"
    JOB_INTERVIEW = "job_interview"
    RESTAURANT = "restaurant"
    AIRPORT = "airport"
    DOCTOR_VISIT = "doctor_visit"
    DEBATE = "debate"
    SMALL_TALK = "small_talk"


class GrammarErrorType(str, Enum):
    TENSE = "tense"
    ARTICLE = "article"
    PREPOSITION = "preposition"
    SUBJECT_VERB_AGREEMENT = "subject_verb_agreement"
    WORD_ORDER = "word_order"
    WORD_CHOICE = "word_choice"
    PUNCTUATION = "punctuation"
    OTHER = "other"


# ──────────────────────────────────────────────────────────────────────────
# Request
# ──────────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The text of the message")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable client-generated session ID")
    transcript: str = Field(..., min_length=1, description="User's latest utterance (already ASR-transcribed on client)")
    history: List[ChatMessage] = Field(default_factory=list, description="Prior conversation turns, oldest first")
    cefr_level: CEFRLevel = Field(default=CEFRLevel.B1, description="Client's current estimated CEFR level")
    role_play_mode: RolePlayMode = Field(default=RolePlayMode.FREE_TALK, description="Active scenario for this session")
    user_profile: Optional[dict] = Field(default=None, description="Optional: native language, goals, etc.")


# ──────────────────────────────────────────────────────────────────────────
# Feedback sub-objects
# ──────────────────────────────────────────────────────────────────────────
class GrammarError(BaseModel):
    type: GrammarErrorType
    original: str
    correction: str
    explanation: str = ""


class GrammarFeedback(BaseModel):
    original: str
    corrected: str
    is_correct: bool
    errors: List[GrammarError] = Field(default_factory=list)


class PronunciationFeedback(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    words: List[dict] = Field(default_factory=list)


class VocabItem(BaseModel):
    word: str
    definition: str = ""
    synonyms: List[str] = Field(default_factory=list)


class VocabularyFeedback(BaseModel):
    new_words: List[VocabItem] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class FeedbackEvent(BaseModel):
    grammar: GrammarFeedback
    pronunciation: PronunciationFeedback
    vocabulary: VocabularyFeedback
    cefr_level: CEFRLevel
    cefr_confidence: float = Field(..., ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────────
# Response
# ──────────────────────────────────────────────────────────────────────────
class ChatResponse(BaseModel):
    session_id: str
    reply: str
    feedback: FeedbackEvent


class StreamEvent(BaseModel):
    type: str
    data: dict = Field(default_factory=dict)