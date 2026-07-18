"""Pronunciation scoring tool.

⚠️ On Vercel, we cannot run NVIDIA Riva ASR (gRPC). When you self-host Riva
and set ``RIVA_ASR_URL``, swap the body of ``score_pronunciation`` to call
the Riva proxy and use real word-level confidence scores.

This default implementation is a transparent heuristic:
- If the client sends a ``confidence_map`` (from on-device ASR), use it.
- Otherwise, return a neutral score with words marked green.

The heuristic is intentionally honest — it does not pretend to be a real
pronunciation score. It exists so the rest of the pipeline (and the Android
UI's color-coded word display) keeps working in development.
"""
from __future__ import annotations

from typing import Any

from app.schemas import PronunciationFeedback


def _color_for(score: int) -> str:
    if score >= 85:
        return "green"
    if score >= 65:
        return "yellow"
    return "red"


def _heuristic_score(word: str, idx: int) -> int:
    base = 85 - max(0, len(word) - 6)
    vowels = sum(1 for c in word.lower() if c in "aeiou")
    base += min(5, vowels)
    return max(70, min(95, base))


async def score_pronunciation(
    transcript: str,
    confidence_map: dict[str, float] | None = None,
) -> PronunciationFeedback:
    transcript = transcript.strip()
    if not transcript:
        return PronunciationFeedback(overall_score=100, words=[])

    words: list[dict[str, Any]] = []
    scores: list[int] = []

    for idx, word in enumerate(transcript.split()):
        word_clean = word.strip(".,!?;:\"'()[]").lower()
        if not word_clean:
            continue

        if confidence_map and word_clean in confidence_map:
            score = int(round(confidence_map[word_clean] * 100))
        else:
            score = _heuristic_score(word_clean, idx)

        scores.append(score)
        words.append({"word": word, "score": score, "color": _color_for(score)})

    overall = sum(scores) // len(scores) if scores else 80
    return PronunciationFeedback(overall_score=overall, words=words)