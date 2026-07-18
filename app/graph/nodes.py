"""LangGraph node implementations.

Node graph:
    analyze  ──▶  classify  ──▶  respond  ──▶  END
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.graph.state import ConversationState
from app.schemas import CEFRLevel, FeedbackEvent
from app.services.nim_llm import (
    get_conversational_chat,
    to_langchain_messages,
)
from app.tools.cefr import classify_cefr
from app.tools.grammar import correct_grammar
from app.tools.pronunciation import score_pronunciation
from app.tools.vocab import extract_vocabulary

log = logging.getLogger(__name__)


ROLE_PLAY_PROMPTS: dict[str, str] = {
    "free_talk": (
        "You are VocaLume, a friendly and encouraging AI English speaking coach. "
        "Have a natural, free-flowing conversation with the learner. Match their "
        "CEFR level — keep your language one notch above theirs so they are "
        "challenged but never lost. Ask follow-up questions to keep them talking. "
        "Never explicitly mention that you are correcting them; corrections appear "
        "in a separate side panel on the learner's screen."
    ),
    "job_interview": (
        "You are a hiring manager conducting a job interview. Ask one interview "
        "question at a time, wait for the candidate's answer, then respond briefly "
        "(acknowledge + ask the next question). Adjust question difficulty to the "
        "candidate's CEFR level. Topics: background, strengths/weaknesses, "
        "projects, teamwork, problem solving. Stay professional but warm."
    ),
    "restaurant": (
        "You are a server at a mid-range restaurant. Greet the customer, take "
        "their order, suggest items, handle questions about ingredients and "
        "allergies, and complete the order. Use natural restaurant English. "
        "Adjust to the customer's CEFR level — be patient with beginners."
    ),
    "airport": (
        "You are an airport check-in and gate agent. Help the traveler with "
        "check-in, baggage, security directions, boarding, and gate changes. "
        "Use clear airport English. Adjust to the traveler's CEFR level."
    ),
    "doctor_visit": (
        "You are a doctor in a general practice clinic. Ask the patient about "
        "their symptoms, duration, severity, and medical history. Give simple "
        "advice and explain next steps. Use plain English. Adjust to the "
        "patient's CEFR level — be reassuring with beginners."
    ),
    "debate": (
        "You are a debate partner. Pick a topic with the learner, take the "
        "opposing view, and exchange arguments. Keep claims clear and ask the "
        "learner to justify theirs. Adjust vocabulary to the learner's CEFR level."
    ),
    "small_talk": (
        "You are a friendly acquaintance making small talk at a social event. "
        "Topics: weather, weekend plans, hobbies, movies, food. Keep exchanges "
        "short (1-2 sentences each). Adjust to the learner's CEFR level."
    ),
}


def _role_play_prompt(role_play_mode: str, cefr_level: CEFRLevel) -> str:
    base = ROLE_PLAY_PROMPTS.get(role_play_mode, ROLE_PLAY_PROMPTS["free_talk"])
    return f"{base}\n\nThe learner's current CEFR level is {cefr_level.value}. Calibrate your vocabulary and sentence complexity accordingly."


async def analyze_node(state: ConversationState) -> dict[str, Any]:
    transcript = state.get("transcript", "")
    cefr_level = state.get("cefr_level", CEFRLevel.B1)
    confidence_map = state.get("confidence_map")

    grammar_result, pronunciation_result, vocab_result = await asyncio.gather(
        correct_grammar(transcript),
        score_pronunciation(transcript, confidence_map),
        extract_vocabulary(transcript),
    )

    feedback = FeedbackEvent(
        grammar=grammar_result,
        pronunciation=pronunciation_result,
        vocabulary=vocab_result,
        cefr_level=cefr_level,
        cefr_confidence=0.5,
    )

    return {"feedback": feedback}


async def classify_node(state: ConversationState) -> dict[str, Any]:
    transcript = state.get("transcript", "")
    prior = state.get("cefr_level", CEFRLevel.B1)

    new_level, confidence = await classify_cefr(transcript, prior_level=prior)

    feedback = state.get("feedback")
    if feedback is not None:
        feedback.cefr_level = new_level
        feedback.cefr_confidence = confidence

    return {"cefr_level": new_level, "feedback": feedback}


async def respond_node(state: ConversationState) -> dict[str, Any]:
    transcript = state.get("transcript", "")
    history = state.get("history", [])
    cefr_level = state.get("cefr_level", CEFRLevel.B1)
    role_play_mode = state.get("role_play_mode")
    mode_str = role_play_mode.value if role_play_mode else "free_talk"

    system_prompt = _role_play_prompt(mode_str, cefr_level)
    messages = to_langchain_messages(system_prompt, history, transcript)

    try:
        chat = get_conversational_chat()
        response = await chat.ainvoke(messages)
        reply = response.content if hasattr(response, "content") else str(response)
        reply = reply.strip() or "Could you say a bit more about that?"
    except Exception as exc:
        log.error("Reply generation failed: %s", exc)
        reply = "I'm sorry, I had trouble generating a response just now. Could you repeat that?"

    return {"reply": reply}