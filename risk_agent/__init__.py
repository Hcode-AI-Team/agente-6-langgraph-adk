"""Risk Specialist Agent — Google ADK A2A server.

Run with:
    python run_risk_agent.py          # starts on port 8080

The ADK agent loader (google.adk.cli.fast_api) discovers agents by importing
each subfolder under `agents_dir` as a package and looking up `root_agent`.
Re-exporting it here makes the agent discoverable without forcing the loader
to inspect agent.py directly.

Agent Card:
    http://127.0.0.1:8080/a2a/risk_agent/.well-known/agent-card.json
"""

from .agent import root_agent

__all__ = ["root_agent"]
