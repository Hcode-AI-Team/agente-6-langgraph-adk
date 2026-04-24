"""Project-wide constants: graph node names and A2A application identifiers."""

from __future__ import annotations

from enum import StrEnum


class Node(StrEnum):
    """Canonical names for every node in the StateGraph.

    Using an enum avoids hard-coded strings scattered across graph.py,
    router.py, and tests — a typo will raise AttributeError instead of
    silently routing to a nonexistent node.
    """

    RAG_RETRIEVAL = "rag_retrieval"
    EVALUATOR = "evaluator"
    RISK_AGENT = "risk_agent"
    SYNTHESIS = "synthesis"
    SUMMARIZER = "summarizer"


# Stable identifiers used when creating sessions with the ADK runner.
# Changing these values would invalidate any persisted session history.
APP_NAME_A2A = "bv_credit_orchestrator"
USER_ID_A2A = "orchestrator"
