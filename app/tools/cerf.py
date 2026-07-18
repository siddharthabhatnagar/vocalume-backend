"""CEFR level classifier.

Few-shot prompted classifier that estimates the CEFR level (A1–C2) of a
single learner utterance. The output is smoothed against the client's
current level so a single short or unusually simple utterance does not
cause wild swings in difficulty.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import CEFRLevel
from app.services.nim_llm import get_analyzer_chat

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert English proficiency rater following the Common European Framework of Reference (CEFR).

You will receive ONE learner utterance. Estimate the speaker's CEFR level
(A1, A2, B1, B2, C1, or C2) based on vocabulary range, grammatical accuracy,
syntactic complexity, and fluency of expression.

Output ONLY a JSON object — no markdown fences, no commentary:

{
  "level": "<one of: A1, A2, B1, B2, C1, C2>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one short sentence>"
}

Examples:
- "Hello, my name is John. I am student." → {"level": "A2", "confidence": 0.85, "reasoning": "Simple sentence structure with article error."}
- "I've been working on this project for two years, which has given me considerable insight into the domain." → {"level": "C1", "confidence": 0.9, "reasoning": "Present perfect continuous, relative clause, sophisticated vocabulary."}
- "The thing is, like, you know, it depends." → {"level": "B1", "confidence": 0.6, "reasoning": "Conversational fillers, no complex grammar."}
"""

VALID_LEVELS = {level.value for level in CEFRLevel}
LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _safe_parse(raw: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _smooth_level(observed: CEFRLevel, observed_confidence: float, prior: CEFRLevel) -> tuple[CEFRLevel, float]:
    if observed_confidence >= 0.8:
        return observed, observed_confidence

    obs_idx = LEVEL_ORDER.index(observed.value)
    pri_idx = LEVEL_ORDER.index(prior.value)
    diff = obs_idx - pri_idx

    if abs(diff) > 1:
        new_idx = pri_idx + (1 if diff > 0 else -1)
        smoothed = CEFRLevel(LEVEL_ORDER[new_idx])
    else:
        smoothed = observed

    final_confidence = max(0.5, observed_confidence)
    return smoothed, final_confidence


async def classify_cefr(transcript: str, prior_level: CEFRLevel = CEFRLevel.B1) -> tuple[CEFRLevel, float]:
    transcript = transcript.strip()
    if not transcript:
        return prior_level, 0.5

    chat = get_analyzer_chat()
    try:
        response = await chat.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f'Utterance: "{transcript}"')]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        data = _safe_parse(raw)

        level_str = str(data.get("level", prior_level.value)).upper()
        if level_str not in VALID_LEVELS:
            return prior_level, 0.5

        observed = CEFRLevel(level_str)
        confidence = float(data.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))

        return _smooth_level(observed, confidence, prior_level)
    except Exception as exc:
        log.warning("CEFR classification failed: %s", exc)
        return prior_level, 0.5