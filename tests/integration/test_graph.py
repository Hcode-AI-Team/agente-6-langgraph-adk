"""Integration tests for the full LangGraph StateGraph.

MOCKING STRATEGY
----------------
Integration tests verify the GRAPH FLOW (correct routing, state propagation)
not the internals of individual nodes.  We therefore mock the four node
functions in the namespace where they are USED — `src.graph` — which is the
module that imports them and passes them to `add_node`.

Why `src.graph.*` and not `src.nodes.evaluator.*`?
LangGraph compiles the graph by capturing the function OBJECTS passed to
`add_node`.  When `mocker.patch("src.graph.node_evaluator", …)` is active
and `build_graph()` is called afterwards (we clear `get_app`'s lru_cache
before each test), `build_graph` picks up the patched object.  Patching
inside `src.nodes.evaluator` would change the original module's binding but
not the local binding that `graph.py` imported at module-load time.

The exception is `query_risk_agent` (ADK A2A client), which `node_risk_agent`
looks up at call time, so we patch it in `src.nodes.risk_agent`.
"""

from __future__ import annotations

import pytest

from src.graph import build_graph, get_app, run
from src.state import OrchestratorState, RAGDocument

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
    """Replace node_rag_retrieval with a stub that returns fixed documents."""
    chosen = docs if docs is not None else _SAMPLE_DOCS
    return mocker.patch(
        "src.graph.node_rag_retrieval",
        return_value={"rag_context": chosen},
    )


def _patch_evaluator(mocker, requires_escalation: bool):
    """Replace node_evaluator with a stub that returns a predetermined decision."""
    return mocker.patch(
        "src.graph.node_evaluator",
        return_value={
            "requires_risk_assessment": requires_escalation,
            "evaluator_rationale": "Mocked rationale for testing.",
        },
    )


def _patch_synthesis(mocker, response: str = "Mocked final response."):
    """Replace node_synthesis with a stub that returns a fixed final response."""
    return mocker.patch(
        "src.graph.node_synthesis",
        return_value={"final_response": response},
    )


def _patch_risk_agent(mocker, response: str = "Mocked risk assessment."):
    """Patch the A2A client call inside node_risk_agent (not the node itself).

    We leave node_risk_agent un-mocked so the real node logic (payload
    construction, error handling) is exercised; only the network call is
    stubbed out.
    """
    return mocker.patch(
        "src.nodes.risk_agent.query_risk_agent",
        return_value=response,
    )


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

    def test_entry_point_is_rag_retrieval(self):
        g = build_graph()
        compiled = g.compile()
        graph_def = compiled.get_graph()
        start_edges = [e for e in graph_def.edges if e.source == "__start__"]
        assert any(e.target == "rag_retrieval" for e in start_edges)

    def test_synthesis_has_edge_to_end(self):
        g = build_graph()
        graph_def = g.compile().get_graph()
        synthesis_edges = [e for e in graph_def.edges if e.source == "synthesis"]
        assert any(e.target == "__end__" for e in synthesis_edges)


# ---------------------------------------------------------------------------
# Full graph execution — direct synthesis path (no risk agent)
# ---------------------------------------------------------------------------


class TestDirectSynthesisPath:
    @pytest.fixture(autouse=True)
    def _clear_app_cache(self):
        # Clear cache BEFORE the test so that build_graph() is called with the
        # mocks already active (mocks are applied by pytest-mock before the
        # test body runs, but after fixtures have been set up).
        get_app.cache_clear()
        yield
        get_app.cache_clear()

    def test_final_response_populated(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        _patch_synthesis(mocker, response="Policy allows R$5M.")

        state = run("What is the credit limit?")

        assert state.final_response == "Policy allows R$5M."

    def test_risk_assessment_response_is_none(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        _patch_synthesis(mocker)

        state = run("Simple policy question?")

        # node_risk_agent must NOT have been called on this path.
        assert state.risk_assessment_response is None

    def test_rag_context_propagated_to_synthesis(self, mocker):
        _patch_rag(mocker, docs=_SAMPLE_DOCS)
        _patch_evaluator(mocker, requires_escalation=False)
        mock_synthesis = _patch_synthesis(mocker)

        run("Credit limit?")

        # Synthesis must have been called with a state that contains the docs.
        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert len(state_arg.rag_context) == len(_SAMPLE_DOCS)
        assert any("R$5M" in doc.text for doc in state_arg.rag_context)

    def test_question_reaches_synthesis(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=False)
        mock_synthesis = _patch_synthesis(mocker)

        run("My specific question?")

        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert state_arg.question == "My specific question?"


# ---------------------------------------------------------------------------
# Full graph execution — risk agent path
# ---------------------------------------------------------------------------


class TestRiskAgentPath:
    @pytest.fixture(autouse=True)
    def _clear_app_cache(self):
        get_app.cache_clear()
        yield
        get_app.cache_clear()

    def test_risk_assessment_response_populated(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker, response="High risk: defer to committee.")
        _patch_synthesis(mocker)

        state = run("R$50M restructuring?")

        assert state.risk_assessment_response == "High risk: defer to committee."

    def test_final_response_populated_after_risk(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker)
        _patch_synthesis(mocker, response="Based on risk assessment, deny.")

        state = run("Restructuring query")

        assert state.final_response == "Based on risk assessment, deny."

    def test_risk_assessment_forwarded_to_synthesis(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker, response="Conditional approval.")
        mock_synthesis = _patch_synthesis(mocker)

        run("High-value credit query")

        # Synthesis must receive the risk assessment in the state it was called with.
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

        # Graph must complete even when the A2A call fails.
        state = run("Any question")
        assert state.final_response == "Partial answer."
        assert "unavailable" in state.risk_assessment_response.lower()

    def test_evaluator_rationale_in_synthesis_state(self, mocker):
        _patch_rag(mocker)
        _patch_evaluator(mocker, requires_escalation=True)
        _patch_risk_agent(mocker, response="Risk OK.")
        mock_synthesis = _patch_synthesis(mocker, response="answer")

        run("Complex query")

        # Synthesis must have been called — which means graph completed.
        mock_synthesis.assert_called_once()
        # The state passed to synthesis must have the evaluator's rationale.
        state_arg: OrchestratorState = mock_synthesis.call_args[0][0]
        assert state_arg.evaluator_rationale == "Mocked rationale for testing."
