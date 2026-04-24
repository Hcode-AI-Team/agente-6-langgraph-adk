"""Integration tests for the full LangGraph StateGraph.

MOCKING STRATEGY
----------------
Integration tests verify GRAPH FLOW (correct routing, state propagation)
not the internals of individual nodes.  We mock nodes in the `src.graph`
namespace — where they are captured by `build_graph()` / `add_node`.

Checkpointer:
    Tests use `MemorySaver` (in-memory) injected by patching `_get_checkpointer`
    so no SQLite file is created and each test class starts with a clean slate.

Thread IDs:
    Each test passes a unique `thread_id` to `run()`.  Uniqueness prevents
    checkpoint bleed between tests that share the same MemorySaver instance.

Summarizer:
    `node_summarizer` is NOT mocked — it runs unconditionally but exits
    immediately when `len(recent_messages) < threshold` (always true here
    since synthesis is mocked and returns no messages).
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.graph import _get_checkpointer, build_graph, run

# get_app is not cached in the new design (checkpointer is cached separately)
from src.graph import get_app
from src.state import ConversationMessage, OrchestratorState, RAGDocument

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

_SAMPLE_DOCS = [
    RAGDocument(
        id="p1",
        distance=0.1,
        text="Standard credit limit for Corporate is R$5M.",
        metadata={},
    )
]


# ---------------------------------------------------------------------------
# Mock helper functions — patch at the graph namespace level
# ---------------------------------------------------------------------------


def _patch_rag(mocker, docs=None):
    chosen = docs if docs is not None else _SAMPLE_DOCS
    return mocker.patch(
        "src.graph.node_rag_retrieval",
        return_value={"rag_context": chosen},
    )


def _patch_evaluator(mocker, requires_escalation: bool):
    return mocker.patch(
        "src.graph.node_evaluator",
        return_value={
            "requires_risk_assessment": requires_escalation,
            "evaluator_rationale": "Mocked rationale for testing.",
        },
    )


def _patch_synthesis(mocker, response: str = "Mocked final response."):
    return mocker.patch(
        "src.graph.node_synthesis",
        return_value={"final_response": response, "recent_messages": []},
    )


def _patch_risk_agent(mocker, response: str = "Mocked risk assessment."):
    """Patch the A2A client call inside node_risk_agent (not the node itself)."""
    return mocker.patch(
        "src.nodes.risk_agent.query_risk_agent",
        return_value=response,
    )


def _tid() -> str:
    """Generate a unique thread ID so tests don't share checkpoint state."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Graph topology tests (no execution needed)
# ---------------------------------------------------------------------------


class TestGraphTopology:
    def test_all_nodes_present(self):
        g = build_graph()
        graph_def = g.compile().get_graph()
        node_names = {n.name for n in graph_def.nodes.values()}
        assert "rag_retrieval" in node_names
        assert "evaluator" in node_names
        assert "risk_agent" in node_names
        assert "synthesis" in node_names
        assert "summarizer" in node_names

    def test_entry_point_is_rag_retrieval(self):
        g = build_graph()
        graph_def = g.compile().get_graph()
        start_edges = [e for e in graph_def.edges if e.source == "__start__"]
        assert any(e.target == "rag_retrieval" for e in start_edges)

    def test_synthesis_leads_to_summarizer(self):
        g = build_graph()
        graph_def = g.compile().get_graph()
        synthesis_edges = [e for e in graph_def.edges if e.source == "synthesis"]
        assert any(e.target == "summarizer" for e in synthesis_edges)

    def test_summarizer_has_edge_to_end(self):
        g = build_graph()
        graph_def = g.compile().get_graph()
        summarizer_edges = [e for e in graph_def.edges if e.source == "summarizer"]
        assert any(e.target == "__end__" for e in summarizer_edges)


# ---------------------------------------------------------------------------
# Shared fixture: swap SQLite checkpointer for in-memory MemorySaver
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _mem_checkpointer(mocker):
    """Replace the SQLite checkpointer with MemorySaver for all execution tests.

    _get_checkpointer is cached with lru_cache; we clear it before and after so
    each test class gets a fresh MemorySaver (no checkpoint bleed between tests).
    get_app is NOT cached — it calls _get_checkpointer on every invocation.
    """
    _get_checkpointer.cache_clear()
    mocker.patch("src.graph._get_checkpointer", return_value=MemorySaver())
    yield
    _get_checkpointer.cache_clear()


# ---------------------------------------------------------------------------
# Full graph execution — direct synthesis path (no risk agent)
# ---------------------------------------------------------------------------


class TestDirectSynthesisPath:
    @pytest.fixture(autouse=True)
    def _setup(self, _mem_checkpointer):
        pass

    def test_final_response_populated(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        _patch_synthesis(mocker, response="Policy allows R$5M.")

        state = run("What is the credit limit?", _tid())

        assert state.final_response == "Policy allows R$5M."

    def test_risk_assessment_response_is_none(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        _patch_synthesis(mocker)

        state = run("Simple policy question?", _tid())

        assert state.risk_assessment_response is None

    def test_rag_context_propagated_to_synthesis(self, mocker):
        _patch_rag(mocker, docs=_SAMPLE_DOCS)
        _patch_evaluator(mocker, requires_escalation=False)
        mock_synthesis = _patch_synthesis(mocker)

        run("Credit limit?", _tid())

        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert len(state_arg.rag_context) == len(_SAMPLE_DOCS)
        assert any("R$5M" in doc.text for doc in state_arg.rag_context)

    def test_question_reaches_synthesis(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        mock_synthesis = _patch_synthesis(mocker)

        run("My specific question?", _tid())

        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert state_arg.question == "My specific question?"


# ---------------------------------------------------------------------------
# Full graph execution — risk agent path
# ---------------------------------------------------------------------------


class TestRiskAgentPath:
    @pytest.fixture(autouse=True)
    def _setup(self, _mem_checkpointer):
        pass

    def test_risk_assessment_response_populated(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker, response="High risk: defer to committee.")
        _patch_synthesis(mocker)

        state = run("R$50M restructuring?", _tid())

        assert state.risk_assessment_response == "High risk: defer to committee."

    def test_final_response_populated_after_risk(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker)
        _patch_synthesis(mocker, response="Based on risk assessment, deny.")

        state = run("Restructuring query", _tid())

        assert state.final_response == "Based on risk assessment, deny."

    def test_risk_assessment_forwarded_to_synthesis(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker, response="Conditional approval.")
        mock_synthesis = _patch_synthesis(mocker)

        run("High-value credit query", _tid())

        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert state_arg.risk_assessment_response == "Conditional approval."

    def test_a2a_failure_does_not_crash_graph(self, mocker):
        from src.clients.adk_client import A2AClientError

        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        mocker.patch(
            "src.nodes.risk_agent.query_risk_agent",
            side_effect=A2AClientError("Service unavailable"),
        )
        _patch_synthesis(mocker, response="Partial answer.")

        state = run("Any question", _tid())
        assert state.final_response == "Partial answer."
        assert "unavailable" in state.risk_assessment_response.lower()

    def test_evaluator_rationale_in_synthesis_state(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker, response="Risk OK.")
        mock_synthesis = _patch_synthesis(mocker, response="answer")

        run("Complex query", _tid())

        mock_synthesis.assert_called_once()
        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert state_arg.evaluator_rationale == "Mocked rationale for testing."


# ---------------------------------------------------------------------------
# Conversation memory — multi-turn checkpointing
# ---------------------------------------------------------------------------


class TestConversationMemory:
    @pytest.fixture(autouse=True)
    def _setup(self, _mem_checkpointer):
        pass

    def test_second_turn_preserves_conversation_summary(self, mocker):
        """Verify that conversation_summary from a prior turn reaches the evaluator."""
        thread = _tid()

        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        _patch_synthesis(
            mocker,
            response="First answer.",
        )
        # Inject a pre-existing conversation_summary via a second synthesis stub
        # that returns a summary so the evaluator sees it on turn 2.
        mocker.patch(
            "src.graph.node_summarizer",
            return_value={"conversation_summary": "Gestor foca em middle market."},
        )

        run("First question", thread)

        # Turn 2: evaluator should receive the summary injected by the summarizer.
        mock_evaluator = _patch_evaluator(mocker, requires_escalation=False)
        _patch_synthesis(mocker, response="Second answer.")

        run("Second question", thread)

        state_arg: OrchestratorState = mock_evaluator.call_args[0][0]
        assert state_arg.conversation_summary == "Gestor foca em middle market."

    def test_question_updates_on_each_turn(self, mocker):
        """Each new invocation with the same thread overwrites the question field."""
        thread = _tid()

        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        _patch_synthesis(mocker)

        run("First question", thread)
        mock_synthesis = _patch_synthesis(mocker)
        run("Second question", thread)

        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert state_arg.question == "Second question"
