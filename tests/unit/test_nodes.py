"""Unit tests for the four node functions.

External dependencies (Vector Search, LLM, ADK A2A) are all mocked so these
tests run without GCP credentials or network access.

MOCKING STRATEGY FOR LangChain CHAINS
--------------------------------------
In node_evaluator and node_synthesis, the code builds a chain via:
    chain = prompt | get_llm().with_structured_output(...)

`prompt` is a MagicMock (we patch ChatPromptTemplate.from_messages).
Python evaluates `prompt | x` as `type(prompt).__or__(prompt, x)`.
For MagicMock, magic methods are configured on the instance's unique class,
so the correct way to set the __or__ return value is:
    mock_prompt.__or__.return_value = mock_chain
NOT:
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)   ← unreliable
The first form configures the child-mock that MagicMock already created for
__or__, ensuring the class-level descriptor calls the right mock.
"""

from __future__ import annotations

import pytest

from src.models import A2AIntent, RiskAssessment
from src.nodes.evaluator import node_evaluator
from src.nodes.rag_retrieval import node_rag_retrieval
from src.nodes.risk_agent import node_risk_agent
from src.nodes.synthesis import node_synthesis
from src.state import OrchestratorState, RAGDocument


# ---------------------------------------------------------------------------
# node_rag_retrieval
# ---------------------------------------------------------------------------


class TestNodeRagRetrieval:
    def test_returns_rag_context_key(self, mocker, base_state):
        mock_docs = [RAGDocument(id="p1", distance=0.1, text="Policy text")]
        mocker.patch("src.nodes.rag_retrieval.search_policies", return_value=mock_docs)

        result = node_rag_retrieval(base_state)

        assert "rag_context" in result
        assert result["rag_context"] == mock_docs

    def test_passes_question_to_search_policies(self, mocker, base_state):
        mock_search = mocker.patch(
            "src.nodes.rag_retrieval.search_policies", return_value=[]
        )
        node_rag_retrieval(base_state)
        mock_search.assert_called_once_with(base_state.question)

    def test_empty_result_is_valid(self, mocker, base_state):
        mocker.patch("src.nodes.rag_retrieval.search_policies", return_value=[])
        result = node_rag_retrieval(base_state)
        assert result["rag_context"] == []


# ---------------------------------------------------------------------------
# Helpers for LangChain chain mocking
# ---------------------------------------------------------------------------


def _make_evaluator_chain_mock(mocker, assessment: RiskAssessment):
    """Return a mock_chain whose .invoke() returns the given RiskAssessment.

    We patch ChatPromptTemplate.from_messages to return a MagicMock prompt,
    then configure mock_prompt.__or__.return_value = mock_chain so that
    `prompt | llm.with_structured_output(...)` yields mock_chain.
    Using __or__.return_value (not __or__ = …) is the reliable way to set
    the return value of a magic-method mock in unittest.mock.
    """
    mock_chain = mocker.MagicMock()
    mock_chain.invoke.return_value = assessment

    mock_prompt = mocker.MagicMock()
    # __or__.return_value configures the existing child mock that MagicMock
    # created for __or__, keeping it consistent with the class-level descriptor.
    mock_prompt.__or__.return_value = mock_chain

    mocker.patch(
        "src.nodes.evaluator.ChatPromptTemplate.from_messages",
        return_value=mock_prompt,
    )
    mocker.patch("src.nodes.evaluator.get_llm")
    return mock_chain


def _make_synthesis_chain_mock(mocker, response_text: str):
    """Return a mock_chain whose .invoke() returns an object with .content."""
    mock_response = mocker.MagicMock()
    mock_response.content = response_text

    mock_chain = mocker.MagicMock()
    mock_chain.invoke.return_value = mock_response

    mock_prompt = mocker.MagicMock()
    mock_prompt.__or__.return_value = mock_chain

    mocker.patch(
        "src.nodes.synthesis.ChatPromptTemplate.from_messages",
        return_value=mock_prompt,
    )
    mocker.patch("src.nodes.synthesis.get_llm")
    return mock_chain


# ---------------------------------------------------------------------------
# node_evaluator
# ---------------------------------------------------------------------------


class TestNodeEvaluator:
    def test_no_escalation_path(self, mocker, state_with_rag):
        assessment = RiskAssessment(requires_escalation=False, rationale="RAG covers this.")
        _make_evaluator_chain_mock(mocker, assessment)

        result = node_evaluator(state_with_rag)

        assert result["requires_risk_assessment"] is False
        assert result["evaluator_rationale"] == "RAG covers this."

    def test_escalation_path(self, mocker, state_with_rag):
        assessment = RiskAssessment(
            requires_escalation=True,
            rationale="Operation exceeds standard limits.",
        )
        _make_evaluator_chain_mock(mocker, assessment)

        result = node_evaluator(state_with_rag)

        assert result["requires_risk_assessment"] is True
        assert "limits" in result["evaluator_rationale"]

    def test_chain_invoked_with_correct_keys(self, mocker, state_with_rag):
        # The chain must be called with both "question" and "context" keys.
        assessment = RiskAssessment(requires_escalation=False, rationale="Sufficient.")
        mock_chain = _make_evaluator_chain_mock(mocker, assessment)

        node_evaluator(state_with_rag)

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "question" in call_kwargs
        assert "context" in call_kwargs

    def test_empty_rag_context_label_in_prompt(self, mocker, base_state):
        # When no documents were retrieved, the context string must signal
        # absence so the LLM knows coverage is insufficient.
        assessment = RiskAssessment(requires_escalation=False, rationale="No docs but OK.")
        mock_chain = _make_evaluator_chain_mock(mocker, assessment)

        node_evaluator(base_state)

        context_value = mock_chain.invoke.call_args[0][0]["context"]
        assert "no documents" in context_value.lower()

    def test_question_passed_verbatim_to_chain(self, mocker, state_with_rag):
        assessment = RiskAssessment(requires_escalation=False, rationale="Fine.")
        mock_chain = _make_evaluator_chain_mock(mocker, assessment)

        node_evaluator(state_with_rag)

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["question"] == state_with_rag.question


# ---------------------------------------------------------------------------
# node_risk_agent
# ---------------------------------------------------------------------------


class TestNodeRiskAgent:
    def test_returns_risk_assessment_response(self, mocker, state_needs_risk):
        mocker.patch(
            "src.nodes.risk_agent.query_risk_agent",
            return_value="Risk assessment: approve with conditions.",
        )
        result = node_risk_agent(state_needs_risk)
        assert result["risk_assessment_response"] == "Risk assessment: approve with conditions."

    def test_payload_contains_question_and_chunks(self, mocker, state_needs_risk):
        mock_query = mocker.patch(
            "src.nodes.risk_agent.query_risk_agent",
            return_value="ok",
        )
        node_risk_agent(state_needs_risk)

        payload = mock_query.call_args[0][0]
        assert payload.question == state_needs_risk.question
        assert len(payload.policy_chunks) == len(state_needs_risk.rag_context)

    def test_payload_intent_is_correct(self, mocker, state_needs_risk):
        mock_query = mocker.patch(
            "src.nodes.risk_agent.query_risk_agent",
            return_value="ok",
        )
        node_risk_agent(state_needs_risk)
        assert mock_query.call_args[0][0].intent == A2AIntent.ASSESS_CREDIT_RISK

    def test_a2a_failure_is_caught_gracefully(self, mocker, state_needs_risk):
        from src.clients.adk_client import A2AClientError

        mocker.patch(
            "src.nodes.risk_agent.query_risk_agent",
            side_effect=A2AClientError("Connection refused"),
        )
        # The node must NOT re-raise; it should return a fallback string.
        result = node_risk_agent(state_needs_risk)
        assert "unavailable" in result["risk_assessment_response"].lower()

    def test_trace_id_is_unique_per_call(self, mocker, state_needs_risk):
        captured_payloads: list = []
        mocker.patch(
            "src.nodes.risk_agent.query_risk_agent",
            side_effect=lambda p: captured_payloads.append(p) or "ok",
        )
        node_risk_agent(state_needs_risk)
        node_risk_agent(state_needs_risk)
        ids = [p.session.trace_id for p in captured_payloads]
        assert ids[0] != ids[1]

    def test_triage_rationale_forwarded_to_payload(self, mocker, state_needs_risk):
        mock_query = mocker.patch(
            "src.nodes.risk_agent.query_risk_agent",
            return_value="ok",
        )
        node_risk_agent(state_needs_risk)
        payload = mock_query.call_args[0][0]
        assert payload.session.triage_rationale == state_needs_risk.evaluator_rationale


# ---------------------------------------------------------------------------
# node_synthesis
# ---------------------------------------------------------------------------


class TestNodeSynthesis:
    def test_returns_final_response(self, mocker, state_with_rag):
        _make_synthesis_chain_mock(mocker, "Final answer for the manager.")
        result = node_synthesis(state_with_rag)
        assert result["final_response"] == "Final answer for the manager."

    def test_prompt_includes_question(self, mocker, state_with_rag):
        mock_chain = _make_synthesis_chain_mock(mocker, "answer")
        node_synthesis(state_with_rag)
        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["question"] == state_with_rag.question

    def test_prompt_includes_not_requested_when_no_assessment(
        self, mocker, state_with_rag
    ):
        # When no A2A call was made, assessment placeholder must be clear.
        mock_chain = _make_synthesis_chain_mock(mocker, "answer")
        node_synthesis(state_with_rag)
        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "not requested" in call_kwargs["assessment"]

    def test_prompt_includes_risk_assessment_when_present(
        self, mocker, state_needs_risk
    ):
        state = state_needs_risk.model_copy(
            update={"risk_assessment_response": "High risk — escalate."}
        )
        mock_chain = _make_synthesis_chain_mock(mocker, "answer")
        node_synthesis(state)
        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["assessment"] == "High risk — escalate."

    def test_empty_rag_context_label(self, mocker):
        state = OrchestratorState(question="q?")
        mock_chain = _make_synthesis_chain_mock(mocker, "answer")
        node_synthesis(state)
        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "no documents" in call_kwargs["context"].lower()

    def test_context_includes_document_text(self, mocker, state_with_rag):
        # RAG docs must be passed to the synthesis prompt so the LLM can cite them.
        mock_chain = _make_synthesis_chain_mock(mocker, "answer")
        node_synthesis(state_with_rag)
        context = mock_chain.invoke.call_args[0][0]["context"]
        # The fixture has a doc with "R$5M" text.
        assert any(
            doc.text in context or doc.text[:30] in context
            for doc in state_with_rag.rag_context
        )
