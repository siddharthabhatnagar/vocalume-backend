"""
Cerebras-backed LLM service wrapper.

Cerebras Inference (https://inference.cerebras.ai/v1) exposes an
OpenAI-compatible `/chat/completions` REST endpoint, so we can reuse
`langchain_openai.ChatOpenAI` unmodified simply by overriding its
`base_url` and `openai_api_key`. `langchain_openai` performs an eager
sanity check for an `OPENAI_API_KEY` in some code paths even when a
key is passed explicitly to the constructor, so we set a harmless
placeholder value via `os.environ.setdefault` at import time, before
`ChatOpenAI` is imported, so construction never fails just because the
real OpenAI key is absent -- only the Cerebras key is ever actually
used for authentication.

This module exposes two cached factory functions for the two LLM
"profiles" the app needs: a low-temperature `analyzer` profile for
structured JSON extraction tasks (grammar, vocabulary, CEFR), and a
higher-temperature `conversational` profile for the free-form role-play
reply. It also provides a small helper to convert wire-format chat
history into LangChain message objects, with history truncation.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "not-needed")

from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache(maxsize=8)
def _cached_chat(model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    """Build (and cache) a ChatOpenAI instance pointed at Cerebras Inference."""
    settings = get_settings()
    if not settings.cerebras_configured:
        raise RuntimeError(
            "CEREBRAS_API_KEY is not configured (expected a key starting with "
            "'csk-'). Set it in your environment or .env file before making "
            "LLM calls."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=settings.cerebras_base_url,
        api_key=settings.cerebras_api_key,
    )


def get_analyzer_chat() -> ChatOpenAI:
    """Return the low-temperature chat model used for structured analysis tasks."""
    settings = get_settings()
    return _cached_chat(settings.cerebras_model, settings.llm_temperature_analyzer, 1024)


def get_conversational_chat() -> ChatOpenAI:
    """Return the higher-temperature chat model used for free-form role-play replies."""
    settings = get_settings()
    return _cached_chat(
        settings.cerebras_model,
        settings.llm_temperature_conversation,
        settings.llm_max_tokens,
    )


def to_langchain_messages(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> list[BaseMessage]:
    """
    Convert a system prompt + wire-format history + latest user message into
    a list of LangChain message objects, truncating history to the most
    recent `max_history_turns * 2` entries (user+assistant pairs) so prompts
    stay bounded in size.
    """
    settings = get_settings()
    max_items = settings.max_history_turns * 2
    trimmed = history[-max_items:] if max_items > 0 else history

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for item in trimmed:
        role = item.get("role")
        content = item.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=user_message))
    return messages
