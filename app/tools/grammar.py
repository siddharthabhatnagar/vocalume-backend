"""
Grammar correction tool.

Uses the Cerebras analyzer chat (low temperature, JSON-only output) to
detect and correct grammar mistakes in a learner's transcript. The
prompt strictly demands a JSON object with no markdown fences and no
commentary, since the Cerebras/Llama model can otherwise wrap its
answer in prose or code fences. Parsing is tolerant: we strip common
code-fence wrappers and, if `json.loads` still fails, fall back to a
regex extraction of the first `{...}` block. If everything fails (bad
JSON, network error, missing API key), we return a no-op
`GrammarFeedback` marking the utterance as correct with zero errors,
since a broken grammar tool should never prevent the rest of the
conversation pipeline from responding to the learner.
"""

import json
import re

from app.schemas import GrammarError, GrammarErrorType, GrammarFeedback
from app.services.cerebras_llm import get_analyzer_chat

_SYSTEM_PROMPT = """You are an English grammar correction engine for a language-learning app.
Given a learner's spoken utterance (transcribed to text), identify grammar mistakes and provide a corrected version.

Output ONLY a JSON object -- no markdown fences, no commentary, no explanation outside the JSON. The JSON must have this exact shape:
{
  "corrected": "<the fully corrected sentence>",
  "is_correct": <true|false>,
  "errors": [
    {
      "type": "<one of: tense, article, preposition, subject_verb_agreement, word_order, word_choice, punctuation, other>",
      "original": "<the incorrect fragment>",
      "correction": "<the corrected fragment>",
      "explanation": "<brief, learner-friendly explanation>"
    }
  ]
}

If the utterance has no grammar errors, return "is_correct": true and an empty "errors" list."""


def _extract_json(raw: str) -> dict:
    """Tolerantly parse a JSON object out of an LLM response string."""
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


async def correct_grammar(transcript: str) -> GrammarFeedback:
    """Run the grammar-correction LLM call and return structured feedback.

    Never raises: on any failure, returns a no-op GrammarFeedback with
    is_correct=True and an empty errors list.
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

        errors = [
            GrammarError(
                type=GrammarErrorType(e.get("type", "other")),
                original=e.get("original", ""),
                correction=e.get("correction", ""),
                explanation=e.get("explanation", ""),
            )
            for e in parsed.get("errors", [])
        ]

        return GrammarFeedback(
            original=transcript,
            corrected=parsed.get("corrected", transcript),
            is_correct=bool(parsed.get("is_correct", len(errors) == 0)),
            errors=errors,
        )
    except Exception:
        return GrammarFeedback(
            original=transcript,
            corrected=transcript,
            is_correct=True,
            errors=[],
        )
