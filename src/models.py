"""Pydantic models for structured I/O: LLM evaluation output and A2A payloads."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .state import RAGDocument


class RiskAssessment(BaseModel):
    """Structured decision returned by the evaluator node via `with_structured_output`.

    Using a Pydantic model here (rather than free-form JSON) guarantees that
    the LLM output is schema-validated before the conditional edge reads it.
    """

    # True forces routing to the ADK risk-agent node.
    requires_escalation: bool = Field(
        ...,
        description=(
            "True when the operation requires specialist risk analysis: "
            "high-value credit, debt restructuring, unusual exposure, "
            "covenants / IFRS 9 / Res. 4.966, or insufficient RAG coverage."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=5,
        description="Concise explanation of the decision, citing the criteria that apply.",
    )


class A2AIntent(StrEnum):
    """Intent codes understood by the remote risk-agent server."""

    ASSESS_CREDIT_RISK = "assess_credit_risk"


class A2ASession(BaseModel):
    """Session metadata propagated by the orchestrator in every A2A call.

    The risk-agent server may use this for tracing, logging, and
    context-aware responses (e.g. knowing which orchestrator invoked it).
    """

    orchestrator: str = Field(default="bv_langgraph")
    trace_id: str
    # Reason why the evaluator decided escalation was necessary.
    triage_rationale: str = ""


class PolicyChunk(BaseModel):
    """One retrieved policy chunk as sent to the risk-agent via A2A.

    We convert RAGDocument → PolicyChunk before serialisation so that
    we control the wire format independently of the internal state model.
    """

    id: str
    distance: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_document(cls, doc: RAGDocument) -> "PolicyChunk":
        """Build a PolicyChunk from a RAGDocument in the orchestrator state."""
        return cls(
            id=doc.id,
            distance=doc.distance,
            text=doc.text,
            metadata=doc.metadata,
        )


class A2APayload(BaseModel):
    """Structured envelope sent to the specialist agent via A2A protocol.

    Serialised to JSON and placed in the `text` field of a `types.Part`.
    Using a dedicated model keeps the wire format explicit and versioned.
    """

    intent: A2AIntent = A2AIntent.ASSESS_CREDIT_RISK
    question: str
    policy_chunks: list[PolicyChunk] = Field(default_factory=list)
    session: A2ASession
