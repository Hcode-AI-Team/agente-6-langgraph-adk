"""Unit tests for src/models.py — A2A payload models and RiskAssessment."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import A2AIntent, A2APayload, A2ASession, PolicyChunk, RiskAssessment
from src.state import RAGDocument


class TestRiskAssessment:
    def test_escalation_true(self):
        ra = RiskAssessment(requires_escalation=True, rationale="High exposure.")
        assert ra.requires_escalation is True

    def test_escalation_false(self):
        ra = RiskAssessment(requires_escalation=False, rationale="RAG sufficient.")
        assert ra.requires_escalation is False

    def test_rationale_minimum_length(self):
        # min_length=5 ensures the LLM always provides a meaningful justification.
        with pytest.raises(ValidationError):
            RiskAssessment(requires_escalation=False, rationale="ok")

    def test_requires_both_fields(self):
        with pytest.raises(ValidationError):
            RiskAssessment(requires_escalation=True)  # type: ignore[call-arg]


class TestA2AIntent:
    def test_enum_value(self):
        assert A2AIntent.ASSESS_CREDIT_RISK == "assess_credit_risk"

    def test_is_str(self):
        # StrEnum makes intent values usable directly as JSON strings.
        assert isinstance(A2AIntent.ASSESS_CREDIT_RISK, str)


class TestPolicyChunk:
    def test_from_document_copies_fields(self):
        doc = RAGDocument(
            id="p1",
            distance=0.07,
            text="Policy text.",
            metadata={"page": "3"},
        )
        chunk = PolicyChunk.from_document(doc)
        assert chunk.id == doc.id
        assert chunk.distance == doc.distance
        assert chunk.text == doc.text
        assert chunk.metadata == doc.metadata

    def test_metadata_defaults_to_empty_dict(self):
        chunk = PolicyChunk(id="c1", distance=0.1, text="x")
        assert chunk.metadata == {}

    def test_direct_creation(self):
        chunk = PolicyChunk(id="c1", distance=0.2, text="some text", metadata={"k": "v"})
        assert chunk.id == "c1"


class TestA2ASession:
    def test_defaults(self):
        session = A2ASession(trace_id="abc123")
        assert session.orchestrator == "bv_langgraph"
        assert session.triage_rationale == ""

    def test_custom_values(self):
        session = A2ASession(
            orchestrator="test_orch",
            trace_id="tid",
            triage_rationale="reason here",
        )
        assert session.orchestrator == "test_orch"
        assert session.triage_rationale == "reason here"

    def test_requires_trace_id(self):
        with pytest.raises(ValidationError):
            A2ASession()  # type: ignore[call-arg]


class TestA2APayload:
    def test_default_intent(self):
        payload = A2APayload(
            question="test?",
            session=A2ASession(trace_id="t1"),
        )
        assert payload.intent == A2AIntent.ASSESS_CREDIT_RISK

    def test_policy_chunks_default_empty(self):
        payload = A2APayload(
            question="test?",
            session=A2ASession(trace_id="t1"),
        )
        assert payload.policy_chunks == []

    def test_serialises_to_json(self):
        # Verify that model_dump_json produces valid JSON without errors.
        payload = A2APayload(
            question="Can I approve R$10M?",
            policy_chunks=[PolicyChunk(id="p1", distance=0.1, text="limit text")],
            session=A2ASession(trace_id="trace-001"),
        )
        json_str = payload.model_dump_json()
        assert "assess_credit_risk" in json_str
        assert "Can I approve" in json_str
        assert "trace-001" in json_str
