"""
Chat endpoints: the core of the VocaLume backend.

`POST /chat` runs the full LangGraph pipeline (analyze -> classify ->
respond) synchronously and returns a single `ChatResponse` containing
both the reply text and the full structured feedback bundle. This is
the simplest integration path for clients that don't need incremental
updates.

`POST /chat/stream` is the latency-optimized path, using Server-Sent
Events (SSE) instead of WebSocket -- Vercel serverless functions don't
support persistent WebSocket connections, but SSE works fine over a
regular HTTP response stream. It first runs just the `analyze` and
`classify` nodes and emits the resulting feedback bundle immediately
(so the Android client can render grammar/pronunciation/vocabulary
feedback in a side panel right away), then streams the conversational
reply token-by-token as it's generated, and finally emits a `done`
event summarizing the turn. Any exception during the stream is caught
and emitted as an `error` event rather than breaking the connection
uncleanly.
"""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.graph.builder import conversation_app
from app.graph.nodes import _role_play_prompt
from app.schemas import ChatRequest, ChatResponse
from app.services.cerebras_llm import get_conversational_chat, to_langchain_messages

router = APIRouter(tags=["chat"])


def _build_initial_state(req: ChatRequest) -> dict:
    """Translate an incoming ChatRequest into the initial ConversationState dict."""
    confidence_map = None
    if req.user_profile:
        confidence_map = req.user_profile.get("confidence_map")

    return {
        "session_id": req.session_id,
        "transcript": req.transcript,
        "history": [m.model_dump() for m in req.history],
        "cefr_level": req.cefr_level,
        "role_play_mode": req.role_play_mode,
        "user_profile": req.user_profile,
        "confidence_map": confidence_map,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Run the full conversation pipeline and return reply + feedback in one shot."""
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript must not be empty")

    state = _build_initial_state(req)
    result = await conversation_app.ainvoke(state)

    feedback = result.get("feedback")
    reply = result.get("reply")
    if feedback is None or reply is None:
        raise HTTPException(status_code=500, detail="pipeline did not produce a complete result")

    return ChatResponse(session_id=req.session_id, reply=reply, feedback=feedback)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """Stream feedback first, then the reply token-by-token, via SSE."""
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript must not be empty")

    async def event_generator():
        try:
            state = _build_initial_state(req)

            # Run only analyze + classify first, so feedback reaches the
            # client as early as possible. We invoke the node functions
            # directly (rather than a full conversation_app.ainvoke) so we
            # can emit the feedback event before generating the reply.
            from app.graph.nodes import analyze_node, classify_node

            analyze_result = await analyze_node(state)
            state.update(analyze_result)
            classify_result = await classify_node(state)
            state.update(classify_result)

            feedback = state.get("feedback")
            if feedback is not None:
                yield {"event": "feedback", "data": feedback.model_dump_json()}

            role_play_mode = state.get("role_play_mode")
            role_play_mode_str = (
                role_play_mode.value if hasattr(role_play_mode, "value") else str(role_play_mode)
            )
            cefr_level = state.get("cefr_level")
            system_prompt = _role_play_prompt(role_play_mode_str, cefr_level)

            chat_model = get_conversational_chat()
            messages = to_langchain_messages(
                system_prompt=system_prompt,
                history=state.get("history", []),
                user_message=state.get("transcript", ""),
            )

            full_reply = ""
            async for chunk in chat_model.astream(messages):
                text = chunk.content or ""
                if text:
                    full_reply += text
                    yield {"event": "token", "data": {"text": text}}

            yield {
                "event": "done",
                "data": {
                    "session_id": req.session_id,
                    "cefr_level": cefr_level.value if hasattr(cefr_level, "value") else cefr_level,
                    "reply": full_reply,
                },
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to client as SSE error event
            yield {"event": "error", "data": {"message": str(exc)}}

    return EventSourceResponse(event_generator())
