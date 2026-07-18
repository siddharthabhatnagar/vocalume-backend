"""
Pydantic v2 wire-contract schemas for the VocaLume backend.

This module is the single source of truth for every shape that crosses
the HTTP boundary: incoming chat requests from the Android client,
outgoing chat responses, the structured feedback bundle (grammar,
pronunciation, vocabulary, CEFR level), and the SSE streaming event
envelope used by /chat/stream. Keeping all of these in one module
means routers, graph nodes, and tools can import a single consistent
set of types instead of re-declaring ad-hoc dicts, which keeps the
JSON emitted to the Kotlin client stable and predictable.
"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CEFRLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class RolePlayMode(str, Enum):
    free_talk = "free_talk"
    job_interview = "job_interview"
    restaurant = "restaurant"
    airport = "airport"
    doctor_visit = "doctor_visit"
    debate = "debate"
    small_talk = "small_talk"


class GrammarErrorType(str, Enum):
    tense = "tense"
    article = "article"
    preposition = "preposition"
    subject_verb_agreement = "subject_verb_agreement"
    word_order = "word_order"
    word_choice = "word_choice"
    punctuation = "punctuation"
    other = "other"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    transcript: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    cefr_level: CEFRLevel = CEFRLevel.B1
    role_play_mode: RolePlayMode = RolePlayMode.free_talk
    user_profile: Optional[dict[str, Any]] = None


class GrammarError(BaseModel):
    type: GrammarErrorType
    original: str
    correction: str
    explanation: str


class GrammarFeedback(BaseModel):
    original: str
    corrected: str
    is_correct: bool
    errors: list[GrammarError] = Field(default_factory=list)


class PronunciationWordScore(BaseModel):
    word: str
    score: float
    color: Literal["green", "yellow", "red"]


class PronunciationFeedback(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    words: list[PronunciationWordScore] = Field(default_factory=list)


class VocabItem(BaseModel):
    word: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)


class VocabularyFeedback(BaseModel):
    new_words: list[VocabItem] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class FeedbackEvent(BaseModel):
    grammar: GrammarFeedback
    pronunciation: PronunciationFeedback
    vocabulary: VocabularyFeedback
    cefr_level: CEFRLevel
    cefr_confidence: float = Field(ge=0, le=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    feedback: FeedbackEvent


class StreamEvent(BaseModel):
    type: Literal["feedback", "token", "done", "error"]
    data: dict[str, Any]
