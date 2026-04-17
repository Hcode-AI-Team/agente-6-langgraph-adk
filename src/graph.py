"""LangGraph StateGraph definition for the credit-policy orchestrator.

The graph wires four nodes with a conditional edge after the evaluator:

  rag_retrieval → evaluator ──(RAG sufficient)──→ synthesis
                          └──(needs risk check)──→ risk_agent → synthesis

`build_graph()` separates construction from compilation so tests can
inspect the graph topology without invoking any external services.
`get_app()` returns the compiled, executable graph (memoised).
`run()` is a convenience helper for one-shot synchronous invocations.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .constants import Node
from .logging_config import get_logger
from .nodes import (
    node_evaluator,
    node_rag_retrieval,
    node_risk_agent,
    node_synthesis,
    route_after_evaluation,
)
from .state import OrchestratorState

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """Construct the StateGraph without compiling it.

    Keeping build and compile separate allows unit tests to call
    `build_graph().get_graph()` to assert on node/edge topology without
    needing real GCP credentials.
    """
    g = StateGraph(OrchestratorState)

    g.add_node(Node.RAG_RETRIEVAL, node_rag_retrieval)
    g.add_node(Node.EVALUATOR, node_evaluator)
    g.add_node(Node.RISK_AGENT, node_risk_agent)
    g.add_node(Node.SYNTHESIS, node_synthesis)

    g.set_entry_point(Node.RAG_RETRIEVAL)
    g.add_edge(Node.RAG_RETRIEVAL, Node.EVALUATOR)

    # route_after_evaluation is a pure function that reads state.requires_risk_assessment
    # and returns the target node name.  The mapping dict makes valid targets explicit.
    g.add_conditional_edges(
        Node.EVALUATOR,
        route_after_evaluation,
        {
            Node.RISK_AGENT: Node.RISK_AGENT,
            Node.SYNTHESIS: Node.SYNTHESIS,
        },
    )

    g.add_edge(Node.RISK_AGENT, Node.SYNTHESIS)
    g.add_edge(Node.SYNTHESIS, END)

    return g


@lru_cache(maxsize=1)
def get_app() -> CompiledStateGraph:
    """Return the compiled graph (memoised to avoid repeated compilation)."""
    logger.info("Compiling orchestrator StateGraph")
    return build_graph().compile()


def run(question: str) -> OrchestratorState:
    """Execute the graph for a single question and return the final typed state.

    LangGraph's `invoke` may return a plain dict when the state schema is a
    Pydantic model; `model_validate` handles both cases transparently.
    """
    app = get_app()
    result = app.invoke(OrchestratorState(question=question))
    if isinstance(result, OrchestratorState):
        return result
    return OrchestratorState.model_validate(result)
