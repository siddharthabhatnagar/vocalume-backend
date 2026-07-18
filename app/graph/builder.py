"""Assemble the LangGraph state machine.

    START ──▶ analyze ──▶ classify ──▶ respond ──▶ END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import analyze_node, classify_node, respond_node
from app.graph.state import ConversationState


def build_graph():
    g = StateGraph(ConversationState)

    g.add_node("analyze", analyze_node)
    g.add_node("classify", classify_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "classify")
    g.add_edge("classify", "respond")
    g.add_edge("respond", END)

    return g.compile()


conversation_app = build_graph()