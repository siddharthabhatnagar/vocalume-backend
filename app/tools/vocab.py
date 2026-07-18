"""Vocabulary extraction tool."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import VocabItem, VocabularyFeedback
from app.services.nim_llm import get_analyzer_chat

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a vocabulary coach for English learners.

You will receive ONE user utterance. Identify any advanced words, idiomatic
phrases, or collocations the learner used (or could have used). Also suggest
2-3 more idiomatic or natural alternative phrasings of the same idea.

Output ONLY a JSON object with this exact shape — no markdown fences, no commentary:

{
  "new_words": [
    {"word": "<the word or phrase>", "definition": "<short def in context>", "synonyms": ["<syn1>", "<syn2>"]}
  ],
  "suggestions": ["<a more idiomatic alternative phrasing>", "<another alternative>"]
}

Rules:
- Only include words that are genuinely worth learning (CEFR B2+ level).
- If the utterance is too simple to extract anything, return empty arrays.
- Suggestions should preserve the user's meaning, not change it.
- Maximum 5 new_words and 3 suggestions.
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


async def extract_vocabulary(transcript: str) -> VocabularyFeedback:
    transcript = transcript.strip()
    if not transcript:
        return VocabularyFeedback()

    chat = get_analyzer_chat()
    try:
        response = await chat.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f'Utterance: "{transcript}"')]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        data = _safe_parse(raw)

        new_words: list[VocabItem] = []
        for raw_w in (data.get("new_words") or [])[:5]:
            if not isinstance(raw_w, dict) or not raw_w.get("word"):
                continue
            new_words.append(
                VocabItem(
                    word=raw_w["word"],
                    definition=raw_w.get("definition", ""),
                    synonyms=raw_w.get("synonyms", []) or [],
                )
            )

        suggestions = [s for s in (data.get("suggestions") or []) if isinstance(s, str)][:3]
        return VocabularyFeedback(new_words=new_words, suggestions=suggestions)
    except Exception as exc:
        log.warning("Vocabulary extraction failed: %s", exc)
        return VocabularyFeedback()