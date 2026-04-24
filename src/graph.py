"""LangGraph StateGraph definition for the credit-policy orchestrator.

Graph topology (5 nodes):

  rag_retrieval → evaluator ──(RAG sufficient)──→ synthesis ──→ summarizer → END
                           └──(needs risk check)──→ risk_agent ──┘

Key additions vs. the original 4-node design:

  Checkpointer (SQLite):
    `get_app()` compiles the graph with a SqliteSaver checkpointer.  Every
    invocation with the same `thread_id` config key loads the previous state
    from the DB, enabling multi-turn conversation continuity.  Invoke with
    `{"question": new_question}` (partial dict) so that LangGraph applies the
    new question as a state update while preserving all other checkpoint fields
    (recent_messages, conversation_summary, etc.).

  Summarizer node:
    Runs unconditionally after synthesis but exits immediately when the
    conversation is below the summarization threshold.  When the threshold is
    reached it consolidates episodic memory (recent_messages) into a procedural
    behavioral profile (conversation_summary) and prunes the message list.

`build_graph()` stays checkpointer-free so topology tests can run without a DB.
`get_app()` is NOT memoised with lru_cache because the SqliteSaver holds a live
DB connection that must be managed explicitly; instead we cache the checkpointer
instance separately and reuse it across `get_app()` calls.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import get_settings
from .constants import Node
from .logging_config import get_logger
from .nodes import (
    node_evaluator,
    node_rag_retrieval,
    node_risk_agent,
    node_summarizer,
    node_synthesis,
    route_after_evaluation,
)
from .state import OrchestratorState

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """Construct the StateGraph without compiling it.

    Keeping build and compile separate allows unit tests to call
    `build_graph().get_graph()` to assert on node/edge topology without
    needing real GCP credentials or a live checkpointer.
    """
    g = StateGraph(OrchestratorState)

    g.add_node(Node.RAG_RETRIEVAL, node_rag_retrieval)
    g.add_node(Node.EVALUATOR, node_evaluator)
    g.add_node(Node.RISK_AGENT, node_risk_agent)
    g.add_node(Node.SYNTHESIS, node_synthesis)
    g.add_node(Node.SUMMARIZER, node_summarizer)

    g.set_entry_point(Node.RAG_RETRIEVAL)
    g.add_edge(Node.RAG_RETRIEVAL, Node.EVALUATOR)

    g.add_conditional_edges(
        Node.EVALUATOR,
        route_after_evaluation,
        {
            Node.RISK_AGENT: Node.RISK_AGENT,
            Node.SYNTHESIS: Node.SYNTHESIS,
        },
    )

    g.add_edge(Node.RISK_AGENT, Node.SYNTHESIS)
    # Summarizer always runs after synthesis; exits early when below threshold.
    g.add_edge(Node.SYNTHESIS, Node.SUMMARIZER)
    g.add_edge(Node.SUMMARIZER, END)

    return g


@lru_cache(maxsize=1)
def _get_checkpointer() -> SqliteSaver:
    """Create and cache the SQLite checkpointer for the process lifetime.

    `SqliteSaver.from_conn_string()` returns a context manager, which LangGraph's
    `compile()` rejects. Instead we open a long-lived `sqlite3.Connection` and
    instantiate `SqliteSaver` directly so it satisfies `BaseCheckpointSaver`.
    `check_same_thread=False` allows LangGraph workers to share the connection.
    """
    settings = get_settings()
    logger.info("Initialising SQLite checkpointer at %s", settings.checkpointer_db_path)
    conn = sqlite3.connect(settings.checkpointer_db_path, check_same_thread=False)
    return SqliteSaver(conn)


def get_app() -> CompiledStateGraph:
    """Return the compiled graph with the SQLite checkpointer attached."""
    logger.info("Compiling orchestrator StateGraph with checkpointer")
    return build_graph().compile(checkpointer=_get_checkpointer())


def run(question: str, thread_id: str) -> OrchestratorState:
    """Execute the graph for one question within a persistent conversation thread.

    Passing a partial dict `{"question": question}` (rather than a full
    OrchestratorState) is intentional: LangGraph merges this update with the
    checkpoint, leaving recent_messages and conversation_summary intact from the
    previous turn.  On the first call for a new thread_id there is no checkpoint,
    so LangGraph initialises all fields from their defaults.
    """
    app = get_app()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    if isinstance(result, OrchestratorState):
        return result
    return OrchestratorState.model_validate(result)
