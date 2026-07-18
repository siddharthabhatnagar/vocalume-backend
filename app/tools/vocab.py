"""
Vocabulary extraction tool.

Uses the Cerebras analyzer chat to surface vocabulary-building
opportunities from a learner's transcript: notable/advanced words the
learner used correctly (worth reinforcing with a definition and
synonyms), plus a short list of suggested alternative words or phrases
the learner could use to sound more natural or precise. As with the
grammar tool, the prompt strictly demands JSON-only output and parsing
is tolerant of markdown code fences or minor formatting noise. Any
failure (bad JSON, LLM/network error) degrades gracefully to an empty
`VocabularyFeedback` rather than raising, so the conversation pipeline
keeps flowing even if this tool has a transient failure.
"""

import json
import re

from app.schemas import VocabItem, VocabularyFeedback
from app.services.cerebras_llm import get_analyzer_chat

_SYSTEM_PROMPT = """You are a vocabulary-building assistant for an English-learning app.
Given a learner's spoken utterance (transcribed to text), identify up to 5 notable words the learner used well (worth reinforcing), and suggest up to 3 alternative words or phrases that could make their speech more natural or precise.

Output ONLY a JSON object -- no markdown fences, no commentary. The JSON must have this exact shape:
{
  "new_words": [
    {"word": "<word>", "definition": "<short, simple definition>", "synonyms": ["<synonym1>", "<synonym2>"]}
  ],
  "suggestions": ["<alternative word or phrase suggestion>"]
}

Limit new_words to at most 5 items and suggestions to at most 3 items. If nothing notable stands out, return empty lists."""


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


async def extract_vocabulary(transcript: str) -> VocabularyFeedback:
    """Run the vocabulary-extraction LLM call and return structured feedback.

    Never raises: on any failure, returns an empty VocabularyFeedback.
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

        new_words = [
            VocabItem(
                word=w.get("word", ""),
                definition=w.get("definition", ""),
                synonyms=list(w.get("synonyms", []))[:5],
            )
            for w in parsed.get("new_words", [])[:5]
        ]
        suggestions = [str(s) for s in parsed.get("suggestions", [])[:3]]

        return VocabularyFeedback(new_words=new_words, suggestions=suggestions)
    except Exception:
        return VocabularyFeedback(new_words=[], suggestions=[])
