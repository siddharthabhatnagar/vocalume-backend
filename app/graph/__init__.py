"""
LangGraph conversation pipeline package.

Defines the `ConversationState` TypedDict shared across nodes, the
three async pipeline nodes (analyze, classify, respond), and the
compiled StateGraph (`conversation_app`) that wires them together as
START -> analyze -> classify -> respond -> END. The graph is compiled
once at import time in `builder.py` so request handlers can reuse the
same compiled graph object across invocations without recompiling it
on every call.
"""
