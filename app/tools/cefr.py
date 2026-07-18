"""
CEFR level classification tool.

Classifies a learner's transcript against the CEFR scale (A1-C2) using
a few-shot-prompted Cerebras analyzer chat call. Three worked examples
(spanning A2, B1, and C1) are embedded directly in the system prompt to
anchor the model's calibration, since raw CEFR labels are otherwise
quite subjective for an LLM to assign consistently.

A single short utterance is a noisy signal, so we apply smoothing
against the learner's prior (previously classified) CEFR level: if the
model reports high confidence (>= 0.8), we trust the fresh observation
outright. Otherwise, we allow the level to drift by at most one CEFR
step toward the observed level from the prior, which prevents a single
unusually simple or complex sentence from causing a wild jump (e.g.
B1 -> C2) in the learner's tracked level.
"""

import json
import re

from app.schemas import CEFRLevel
from app.services.cerebras_llm import get_analyzer_chat

_LEVEL_ORDER = [
    CEFRLevel.A1,
    CEFRLevel.A2,
    CEFRLevel.B1,
    CEFRLevel.B2,
    CEFRLevel.C1,
    CEFRLevel.C2,
]

_SYSTEM_PROMPT = """You are a CEFR (Common European Framework of Reference) level classifier for English learners.
Classify the learner's utterance into one of: A1, A2, B1, B2, C1, C2.

Examples:
Utterance: "My name is Ana. I am from Spain. I like cats."
{"level": "A2", "confidence": 0.85, "reasoning": "Simple present tense, basic vocabulary, short sentences."}

Utterance: "I've been working at this company for about three years now, and honestly, I think I've learned a lot, especially about how to manage a team."
{"level": "B1", "confidence": 0.75, "reasoning": "Present perfect continuous, some complex clauses, everyday vocabulary, generally fluent."}

Utterance: "Although the proposal has considerable merit, I'm skeptical that it adequately addresses the underlying structural issues that have plagued the department for years."
{"level": "C1", "confidence": 0.9, "reasoning": "Complex subordination, precise abstract vocabulary, nuanced argumentation."}

Output ONLY a JSON object -- no markdown fences, no commentary. Shape:
{"level": "<A1|A2|B1|B2|C1|C2>", "confidence": <0.0-1.0>, "reasoning": "<brief reasoning>"}"""


def _extract_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _smooth(observed: CEFRLevel, confidence: float, prior: CEFRLevel) -> CEFRLevel:
    """Smooth the observed level toward the prior unless confidence is high."""
    if confidence >= 0.8:
        return observed

    prior_idx = _LEVEL_ORDER.index(prior)
    observed_idx = _LEVEL_ORDER.index(observed)
    if observed_idx == prior_idx:
        return observed

    step = 1 if observed_idx > prior_idx else -1
    smoothed_idx = prior_idx + step
    smoothed_idx = max(0, min(len(_LEVEL_ORDER) - 1, smoothed_idx))
    return _LEVEL_ORDER[smoothed_idx]


async def classify_cefr(
    transcript: str,
    prior_level: CEFRLevel = CEFRLevel.B1,
) -> tuple[CEFRLevel, float]:
    """Classify the transcript's CEFR level, smoothed against the prior level.

    Never raises: on any failure, returns (prior_level, 0.3) -- i.e. keep
    the learner's existing level with low confidence rather than crash.
    """
    try:
        chat = get_analyzer_chat()
        response = await chat.ainvoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ]
        )
        parsed = _extract_json(response.content)

        observed = CEFRLevel(parsed.get("level", prior_level.value))
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        final_level = _smooth(observed, confidence, prior_level)
        return final_level, confidence
    except Exception:
        return prior_level, 0.3
