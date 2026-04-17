"""Unit tests for src/state.py — RAGDocument and OrchestratorState models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.state import OrchestratorState, RAGDocument


class TestRAGDocument:
    def test_creation_with_required_fields(self):
        doc = RAGDocument(id="d1", distance=0.05, text="Policy text here.", metadata={})
        assert doc.id == "d1"
        assert doc.distance == 0.05
        assert doc.text == "Policy text here."

    def test_metadata_defaults_to_empty_dict(self):
        # default_factory produces a fresh dict per instance — not a shared object.
        doc1 = RAGDocument(id="d1", distance=0.0, text="a")
        doc2 = RAGDocument(id="d2", distance=0.0, text="b")
        assert doc1.metadata == {}
        assert doc1.metadata is not doc2.metadata

    def test_is_immutable(self):
        # frozen=True — pydantic v2 raises ValidationError on any field mutation.
        doc = RAGDocument(id="d1", distance=0.1, text="x")
        with pytest.raises(ValidationError):
            doc.id = "other"  # type: ignore[misc]

    def test_distance_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            RAGDocument(id="d1", distance=-0.01, text="x")

    def test_preview_returns_full_text_when_short(self):
        doc = RAGDocument(id="d1", distance=0.0, text="short text")
        assert doc.preview() == "short text"

    def test_preview_truncates_long_text(self):
        long_text = "a" * 300
        doc = RAGDocument(id="d1", distance=0.0, text=long_text)
        result = doc.preview(max_chars=50)
        # Truncated result must be exactly max_chars long and end with "…"
        assert len(result) == 50
        assert result.endswith("...")

    def test_preview_collapses_newlines(self):
        # Newlines are replaced with spaces so prompts stay single-line.
        doc = RAGDocument(id="d1", distance=0.0, text="line one\nline two")
        assert "\n" not in doc.preview()

    def test_preview_exact_boundary(self):
        # Text of exactly max_chars must NOT be truncated.
        text = "x" * 10
        doc = RAGDocument(id="d1", distance=0.0, text=text)
        assert doc.preview(max_chars=10) == text
        assert not doc.preview(max_chars=10).endswith("...")

    def test_preview_default_max_chars(self):
        # Default truncation length is 240.
        exactly_240 = "z" * 240
        doc = RAGDocument(id="d1", distance=0.0, text=exactly_240)
        assert doc.preview() == exactly_240  # no truncation at exactly 240

        over_240 = "z" * 241
        doc2 = RAGDocument(id="d1", distance=0.0, text=over_240)
        result = doc2.preview()
        assert len(result) == 240
        assert result.endswith("...")


class TestOrchestratorState:
    def test_creation_requires_question(self):
        with pytest.raises(ValidationError):
            OrchestratorState()  # type: ignore[call-arg]

    def test_defaults_are_safe(self):
        state = OrchestratorState(question="test?")
        assert state.rag_context == []
        assert state.requires_risk_assessment is False
        assert state.evaluator_rationale == ""
        assert state.risk_assessment_response is None
        assert state.final_response == ""

    def test_rag_context_is_independent_across_instances(self):
        # Each instance must get its own list; a shared mutable default is a bug.
        s1 = OrchestratorState(question="q1")
        s2 = OrchestratorState(question="q2")
        assert s1.rag_context is not s2.rag_context

    def test_partial_dict_merge_via_model_copy(self):
        # model_copy(update=delta) simulates LangGraph's state-merge step.
        state = OrchestratorState(question="q?")
        delta = {"requires_risk_assessment": True, "evaluator_rationale": "reason"}
        updated = state.model_copy(update=delta)
        assert updated.requires_risk_assessment is True
        assert updated.evaluator_rationale == "reason"
        # Fields not in delta must remain unchanged.
        assert updated.question == "q?"
        assert updated.rag_context == []

    def test_rag_context_accepts_list_of_documents(self):
        docs = [
            RAGDocument(id="p1", distance=0.1, text="Policy A"),
            RAGDocument(id="p2", distance=0.2, text="Policy B"),
        ]
        state = OrchestratorState(question="q?", rag_context=docs)
        assert len(state.rag_context) == 2
        assert state.rag_context[0].id == "p1"
