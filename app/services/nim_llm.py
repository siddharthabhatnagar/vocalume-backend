"""NVIDIA NIM LLM client wrapper.

Uses ``langchain_nvidia_ai_endpoints.ChatNVIDIA`` to talk to NVIDIA's hosted
NIM endpoint at https://integrate.api.nvidia.com/v1. This works on Vercel
because it's plain HTTPS — no gRPC, no persistent connection.
"""
from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import settings


@lru_cache(maxsize=8)
def _build_chat(model: str, temperature: float, max_tokens: int) -> ChatNVIDIA:
    if not settings.nim_configured:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured. Set it in .env (local) or as a "
            "Vercel environment variable (production). Get a key at "
            "https://build.nvidia.com/."
        )
    return ChatNVIDIA(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        nvidia_api_key=settings.nvidia_api_key,
    )


def get_chat(model: str | None = None, temperature: float | None = None, max_tokens: int | None = None) -> ChatNVIDIA:
    return _build_chat(
        model=model or settings.nim_llm_model,
        temperature=temperature if temperature is not None else settings.nim_llm_temperature,
        max_tokens=max_tokens or settings.nim_llm_max_tokens,
    )


def get_analyzer_chat() -> ChatNVIDIA:
    """Lower-temperature model for analytical tasks (grammar, CEFR)."""
    return _build_chat(model=settings.nim_llm_model, temperature=0.1, max_tokens=1024)


def get_conversational_chat() -> ChatNVIDIA:
    """Warmer model for natural conversation replies."""
    return _build_chat(
        model=settings.nim_llm_model,
        temperature=settings.nim_llm_temperature,
        max_tokens=settings.nim_llm_max_tokens,
    )


def to_langchain_messages(system_prompt: str, history: list[dict], user_message: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    trimmed = history[-settings.max_history_turns * 2 :]
    for item in trimmed:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=user_message))
    return messages


async def ainvoke_chat(messages: list[BaseMessage], chat: ChatNVIDIA | None = None) -> str:
    chat = chat or get_conversational_chat()
    response = await chat.ainvoke(messages)
    return response.content if isinstance(response, BaseMessage) else str(response)


async def astream_chat(messages: list[BaseMessage], chat: ChatNVIDIA | None = None) -> AsyncIterator[str]:
    chat = chat or get_conversational_chat()
    async for chunk in chat.astream(messages):
        if isinstance(chunk, BaseMessage) and chunk.content:
            yield chunk.content