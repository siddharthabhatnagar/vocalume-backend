"""
LangGraph pipeline nodes for the VocaLume conversation graph.

Three async nodes make up the pipeline:

- `analyze_node` runs grammar correction, pronunciation scoring, and
  vocabulary extraction concurrently via `asyncio.gather`. This is the
  key latency optimization in the whole backend: these three tools are
  independent of each other, so running them in parallel instead of
  sequentially cuts the analysis phase's wall-clock time down to
  roughly the slowest single call instead of the sum of all three.
- `classify_node` runs CEFR classification (which benefits from
  already having the transcript, and is kept separate from `analyze`
  so it can be skipped independently in some future graph variant).
- `respond_node` builds a role-play-appropriate system prompt
  calibrated to the learner's current CEFR level and calls the
  conversational (higher-temperature) Cerebras chat model to produce
  the actual reply text the learner will hear via TTS.

`ROLE_PLAY_PROMPTS` defines the persona and behavior for each of the
7 supported role-play scenarios. Every persona prompt explicitly
instructs the model never to mention grammar corrections in its reply
text -- corrections are surfaced separately to the Android client as
structured `GrammarFeedback` and rendered in a side panel, so the
conversational reply should read as a natural, in-character response.
"""

import asyncio

from app.graph.state import ConversationState
from app.schemas import CEFRLevel, FeedbackEvent, RolePlayMode
from app.services.cerebras_llm import get_conversational_chat, to_langchain_messages
from app.tools.cefr import classify_cefr
from app.tools.grammar import correct_grammar
from app.tools.pronunciation import score_pronunciation
from app.tools.vocab import extract_vocabulary

ROLE_PLAY_PROMPTS: dict[str, str] = {
    "free_talk": (
        "You are a friendly, encouraging English conversation partner having a "
        "casual, open-ended chat with a learner. Keep the conversation flowing "
        "naturally, ask follow-up questions, and show genuine interest in what "
        "they say."
    ),
    "job_interview": (
        "You are a professional but warm hiring manager conducting a mock job "
        "interview. Ask realistic interview questions one at a time, react "
        "naturally to the candidate's answers, and keep the tone encouraging "
        "even while staying realistic."
    ),
    "restaurant": (
        "You are a friendly waiter/waitress at a restaurant. Greet the customer, "
        "help them order food and drinks, answer questions about the menu, and "
        "respond naturally to their requests, exactly as a real server would."
    ),
    "airport": (
        "You are an airline check-in / gate agent at an airport. Help the "
        "traveler with check-in, baggage questions, boarding information, or "
        "general travel questions, responding naturally and professionally."
    ),
    "doctor_visit": (
        "You are a caring, professional doctor conducting a check-up "
        "consultation. Ask about symptoms, listen to the patient's answers, and "
        "respond the way a real physician would during a routine visit."
    ),
    "debate": (
        "You are a respectful debate partner. Take a clear, reasoned stance on "
        "the topic the learner raises (or propose a topic if none is given), "
        "and challenge their arguments constructively to help them practice "
        "persuasive and argumentative English."
    ),
    "small_talk": (
        "You are a friendly acquaintance making small talk -- at a bus stop, in "
        "a waiting room, or at a casual social gathering. Keep the exchange "
        "light, natural, and easy to follow."
    ),
}

_COMMON_SUFFIX = (
    "\n\nStay fully in character and reply only with what your character would "
    "naturally say -- never mention grammar mistakes, corrections, or scoring "
    "in your reply; those are shown to the learner separately in a side panel. "
    "Keep replies conversational and concise (1-4 sentences) so the exchange "
    "feels like real spoken dialogue."
)


def _role_play_prompt(role_play_mode_str: str, cefr_level: CEFRLevel) -> str:
    """Build the full system prompt for a role-play scenario, calibrated to
    the learner's current CEFR level."""
    base_prompt = ROLE_PLAY_PROMPTS.get(role_play_mode_str, ROLE_PLAY_PROMPTS["free_talk"])
    base_prompt = base_prompt + _COMMON_SUFFIX
    return (
        base_prompt
        + f"\n\nThe learner's current CEFR level is {cefr_level.value}. "
        "Calibrate your vocabulary and sentence complexity accordingly."
    )


async def analyze_node(state: ConversationState) -> dict:
    """Run grammar, pronunciation, and vocabulary analysis concurrently."""
    transcript = state.get("transcript", "")
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
        cefr_level=state.get("cefr_level", CEFRLevel.B1),
        cefr_confidence=0.5,
    )
    return {"feedback": feedback}


async def classify_node(state: ConversationState) -> dict:
    """Classify the learner's CEFR level and fold it into the feedback bundle."""
    transcript = state.get("transcript", "")
    prior_level = state.get("cefr_level", CEFRLevel.B1)

    new_level, confidence = await classify_cefr(transcript, prior_level=prior_level)

    feedback = state.get("feedback")
    if feedback is not None:
        feedback = feedback.model_copy(
            update={"cefr_level": new_level, "cefr_confidence": confidence}
        )

    return {"cefr_level": new_level, "feedback": feedback}


async def respond_node(state: ConversationState) -> dict:
    """Generate the in-character conversational reply via the Cerebras chat model."""
    role_play_mode = state.get("role_play_mode", RolePlayMode.free_talk)
    role_play_mode_str = (
        role_play_mode.value if isinstance(role_play_mode, RolePlayMode) else str(role_play_mode)
    )
    cefr_level = state.get("cefr_level", CEFRLevel.B1)

    system_prompt = _role_play_prompt(role_play_mode_str, cefr_level)

    chat = get_conversational_chat()
    messages = to_langchain_messages(
        system_prompt=system_prompt,
        history=state.get("history", []),
        user_message=state.get("transcript", ""),
    )
    response = await chat.ainvoke(messages)
    return {"reply": response.content}
