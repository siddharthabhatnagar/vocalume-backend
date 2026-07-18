"""Chat endpoints — the core of VocaLume.

- POST /chat            Non-streaming.
- POST /chat/stream     SSE streaming. Emits 'feedback' first, then 'token'
                        chunks of the reply, then 'done'.

Vercel note: WebSocket is not supported on serverless, so we use SSE.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.graph.builder import conversation_app
from app.graph.state import ConversationState
from app.schemas import (
    CEFRLevel,
    ChatRequest,
    ChatResponse,
    FeedbackEvent,
)
from app.services.nim_llm import get_conversational_chat, to_langchain_messages
from app.graph.nodes import _role_play_prompt

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _build_initial_state(req: ChatRequest) -> ConversationState:
    return ConversationState(
        session_id=req.session_id,
        transcript=req.transcript,
        history=[m.model_dump() for m in req.history],
        cefr_level=req.cefr_level,
        role_play_mode=req.role_play_mode,
        user_profile=req.user_profile,
        confidence_map=req.user_profile.get("confidence_map") if req.user_profile else None,
    )


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript must not be empty")

    state = _build_initial_state(req)
    final_state = await conversation_app.ainvoke(state)

    feedback: FeedbackEvent | None = final_state.get("feedback")
    if feedback is None:
        raise HTTPException(status_code=500, detail="analyzer produced no feedback")

    return ChatResponse(
        session_id=req.session_id,
        reply=final_state.get("reply", ""),
        feedback=feedback,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript must not be empty")

    async def event_generator() -> AsyncIterator[dict]:
        try:
            state = _build_initial_state(req)
            state_after_classify = await conversation_app.ainvoke(
                state,
                nodes=["analyze", "classify"],
            )
            feedback: FeedbackEvent | None = state_after_classify.get("feedback")
            if feedback is None:
                yield {
                    "event": "error",
                    "data": json.dumps({"message": "analyzer produced no feedback"}),
                }
                return

            yield {"event": "feedback", "data": feedback.model_dump_json()}

            cefr_level = state_after_classify.get("cefr_level", req.cefr_level)
            role_play_mode = state_after_classify.get("role_play_mode", req.role_play_mode)
            mode_str = role_play_mode.value if role_play_mode else "free_talk"
            system_prompt = _role_play_prompt(mode_str, cefr_level)
            messages = to_langchain_messages(
                system_prompt,
                state_after_classify.get("history", []),
                req.transcript,
            )

            chat = get_conversational_chat()
            full_reply_parts: list[str] = []
            async for chunk in chat.astream(messages):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not text:
                    continue
                full_reply_parts.append(text)
                yield {"event": "token", "data": json.dumps({"text": text})}

            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "session_id": req.session_id,
                        "cefr_level": cefr_level.value if hasattr(cefr_level, "value") else str(cefr_level),
                        "reply": "".join(full_reply_parts),
                    }
                ),
            }
        except Exception as exc:
            log.exception("stream failed")
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_generator())