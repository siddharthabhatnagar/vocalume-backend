"""
LangGraph StateGraph assembly for the VocaLume conversation pipeline.

Wires the three nodes defined in `nodes.py` into a linear graph:

    START -> analyze -> classify -> respond -> END

`analyze` runs grammar/pronunciation/vocabulary analysis in parallel,
`classify` determines (and smooths) the learner's CEFR level, and
`respond` generates the in-character conversational reply calibrated
to that CEFR level. The graph is compiled once at import time into the
`conversation_app` singleton so route handlers can call
`conversation_app.ainvoke(...)` (or `.astream(...)`) without paying a
recompilation cost on every request.
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import analyze_node, classify_node, respond_node
from app.graph.state import ConversationState


def build_graph():
    """Construct and compile the VocaLume conversation StateGraph."""
    graph = StateGraph(ConversationState)

    graph.add_node("analyze", analyze_node)
    graph.add_node("classify", classify_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


conversation_app = build_graph()
