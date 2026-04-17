"""Unit tests for src/nodes/router.py — route_after_evaluation.

route_after_evaluation is a pure function: no mocking needed.
"""

from __future__ import annotations

from src.constants import Node
from src.nodes.router import route_after_evaluation
from src.state import OrchestratorState


class TestRouteAfterEvaluation:
    def test_routes_to_risk_agent_when_escalation_required(self):
        state = OrchestratorState(
            question="Restructuring query",
            requires_risk_assessment=True,
        )
        assert route_after_evaluation(state) == Node.RISK_AGENT

    def test_routes_to_synthesis_when_rag_sufficient(self):
        state = OrchestratorState(
            question="Simple policy query",
            requires_risk_assessment=False,
        )
        assert route_after_evaluation(state) == Node.SYNTHESIS

    def test_default_state_routes_to_synthesis(self):
        # requires_risk_assessment defaults to False, so synthesis is the default path.
        state = OrchestratorState(question="What is the limit?")
        assert route_after_evaluation(state) == Node.SYNTHESIS

    def test_return_type_is_node_enum(self):
        state = OrchestratorState(question="q?", requires_risk_assessment=True)
        result = route_after_evaluation(state)
        assert isinstance(result, Node)

    def test_return_value_matches_graph_mapping_keys(self):
        # The returned values must exactly match the keys used in
        # add_conditional_edges in graph.py — otherwise LangGraph will error.
        for flag, expected_node in [(True, Node.RISK_AGENT), (False, Node.SYNTHESIS)]:
            state = OrchestratorState(question="q?", requires_risk_assessment=flag)
            assert route_after_evaluation(state) == expected_node
