"""Shared pytest fixtures used across unit and integration test modules.

Fixtures here are automatically discovered by pytest without explicit imports.
"""

from __future__ import annotations

import pytest

from src.state import OrchestratorState, RAGDocument


@pytest.fixture()
def sample_document() -> RAGDocument:
    """A single RAGDocument with realistic field values."""
    return RAGDocument(
        id="pol-001",
        distance=0.12,
        text="Standard working-capital credit limit is R$5M for Corporate segment clients.",
        metadata={"source": "credit_policy_2024", "page": "12"},
    )


@pytest.fixture()
def sample_documents(sample_document: RAGDocument) -> list[RAGDocument]:
    """A small list of RAGDocuments simulating a retrieval result."""
    second = RAGDocument(
        id="pol-002",
        distance=0.18,
        text="Clients in workout require approval from the Credit Committee.",
        metadata={"source": "credit_policy_2024", "page": "45"},
    )
    return [sample_document, second]


@pytest.fixture()
def base_state() -> OrchestratorState:
    """Minimal state with only the question set — all other fields at defaults."""
    return OrchestratorState(question="What is the credit limit for Corporate clients?")


@pytest.fixture()
def state_with_rag(sample_documents: list[RAGDocument]) -> OrchestratorState:
    """State that simulates completion of the rag_retrieval node."""
    return OrchestratorState(
        question="What is the credit limit for Corporate clients?",
        rag_context=sample_documents,
    )


@pytest.fixture()
def state_needs_risk(sample_documents: list[RAGDocument]) -> OrchestratorState:
    """State as it would look after the evaluator decided escalation is needed."""
    return OrchestratorState(
        question="Can I approve R$50M restructuring for client X with rating B?",
        rag_context=sample_documents,
        requires_risk_assessment=True,
        evaluator_rationale="Amount exceeds standard limits; restructuring requires specialist review.",
    )
