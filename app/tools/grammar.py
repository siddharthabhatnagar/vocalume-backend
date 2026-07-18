"""Grammar correction tool."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import GrammarError, GrammarErrorType, GrammarFeedback
from app.services.nim_llm import get_analyzer_chat

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict grammar checker for English learners.

You will receive ONE user utterance. Output ONLY a JSON object with this exact
shape — no commentary, no markdown fences:

{
  "corrected": "<the grammatically correct version of the utterance>",
  "is_correct": <true if no correction was needed>,
  "errors": [
    {
      "type": "<one of: tense, article, preposition, subject_verb_agreement, word_order, word_choice, punctuation, other>",
      "original": "<the incorrect fragment, verbatim from the user>",
      "correction": "<the corrected fragment>",
      "explanation": "<one short sentence explaining the fix>"
    }
  ]
}

Rules:
- Preserve the user's meaning and tone. Do not rewrite idiomatic phrases.
- If the user's utterance is already correct, set is_correct=true and errors=[].
- Never invent errors. If you are unsure, do not flag it.
- Keep "corrected" close to the original — fix only what is genuinely wrong.
"""


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


def _validate(data: dict[str, Any], original: str) -> GrammarFeedback:
    if not isinstance(data, dict) or "corrected" not in data:
        return GrammarFeedback(original=original, corrected=original, is_correct=True, errors=[])

    errors: list[GrammarError] = []
    for raw_err in data.get("errors", []) or []:
        try:
            err_type = GrammarErrorType(raw_err.get("type", "other"))
        except ValueError:
            err_type = GrammarErrorType.OTHER
        errors.append(
            GrammarError(
                type=err_type,
                original=raw_err.get("original", ""),
                correction=raw_err.get("correction", ""),
                explanation=raw_err.get("explanation", ""),
            )
        )

    corrected = data.get("corrected", original) or original
    is_correct = bool(data.get("is_correct", corrected == original))

    return GrammarFeedback(original=original, corrected=corrected, is_correct=is_correct, errors=errors)


async def correct_grammar(transcript: str) -> GrammarFeedback:
    transcript = transcript.strip()
    if not transcript:
        return GrammarFeedback(original="", corrected="", is_correct=True, errors=[])

    chat = get_analyzer_chat()
    try:
        response = await chat.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f'Utterance: "{transcript}"')]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _safe_parse(raw)
        return _validate(parsed, transcript)
    except Exception as exc:
        log.warning("Grammar correction failed: %s", exc)
        return GrammarFeedback(original=transcript, corrected=transcript, is_correct=True, errors=[])