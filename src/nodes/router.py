"""Conditional edge function that routes execution after the evaluator node."""

from __future__ import annotations

from ..constants import Node
from ..state import OrchestratorState


def route_after_evaluation(state: OrchestratorState) -> Node:
    """Return the name of the next node based on the evaluator's decision.

    This is a pure function with no side effects — straightforward to unit-test
    without any mocks.  LangGraph calls it after `node_evaluator` completes and
    uses the returned value to look up the target in `add_conditional_edges`.
    """
    return Node.RISK_AGENT if state.requires_risk_assessment else Node.SYNTHESIS
