"""Node 3 (conditional) — Risk agent: delegate to the remote ADK specialist via A2A.

This node is only executed when the evaluator decides that specialist analysis is
required.  It builds a structured `A2APayload`, calls the remote risk-agent, and
stores the response in `risk_assessment_response`.

Graceful degradation: if the A2A call fails (network error, timeout, agent
unavailable), we set a descriptive fallback message instead of letting the
exception propagate.  The synthesis node will then alert the account manager
about the partial result.
"""

from __future__ import annotations

import uuid

from ..clients.adk_client import A2AClientError, query_risk_agent
from ..logging_config import get_logger
from ..models import A2APayload, A2ASession, PolicyChunk
from ..state import OrchestratorState

logger = get_logger(__name__)


def node_risk_agent(state: OrchestratorState) -> dict:
    """Invoke the remote risk-specialist agent and capture its assessment."""
    payload = A2APayload(
        question=state.question,
        # Convert RAGDocuments to PolicyChunks (wire-format model) so that the
        # A2A payload schema is independent of the internal state schema.
        policy_chunks=[PolicyChunk.from_document(doc) for doc in state.rag_context],
        session=A2ASession(
            # uuid4 hex gives a unique trace ID for each orchestrator invocation,
            # enabling end-to-end tracing across the orchestrator and risk-agent.
            trace_id=uuid.uuid4().hex,
            triage_rationale=state.evaluator_rationale,
        ),
    )

    logger.info("node_risk_agent | delegating to risk specialist (A2A)")

    try:
        assessment = query_risk_agent(payload)
    except A2AClientError as exc:
        # Log the error but allow the graph to continue to the synthesis node.
        logger.error("node_risk_agent | A2A call failed: %s", exc)
        assessment = (
            f"[Assessment unavailable — A2A call failed: {exc}]. "
            "Please inform the account manager that specialist review is pending."
        )

    return {"risk_assessment_response": assessment}
