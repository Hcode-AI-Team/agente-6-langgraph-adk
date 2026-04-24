"""Entry point for the Risk Specialist Agent A2A server.

Starts a FastAPI application that exposes the Google ADK agent as an
Agent-to-Agent (A2A) endpoint.  The LangGraph orchestrator connects to
this server via RemoteA2aAgent using the Agent Card URL.

Usage:
    python run_risk_agent.py                 # default host/port 127.0.0.1:8080
    python run_risk_agent.py --port 9090     # custom port
    python run_risk_agent.py --host 0.0.0.0  # expose on all interfaces

Endpoints exposed by the ADK server (when `a2a=True`):
    GET  /a2a/risk_agent/.well-known/agent-card.json   Agent Card (A2A discovery)
    POST /a2a/risk_agent                                JSON-RPC endpoint
    POST /run                                           ADK generic runner (sync)
    POST /run_sse                                       ADK generic runner (SSE)

Notes on ADK 1.31 behaviour:
  * `get_fast_api_app` takes `agents_dir` (plural) — the *parent* directory
    that contains one subfolder per agent.  Each subfolder must contain:
        - `__init__.py` re-exporting `root_agent`
        - `agent.py`    defining `root_agent`
        - `agent.json`  static Agent Card (required for A2A discovery)
  * When `a2a=True`, the ADK auto-registers a2a routes under `/a2a/<folder>`.
  * `Path.cwd() / agents_dir` is used internally, so we must `chdir` to the
    project root before calling `get_fast_api_app`.

Set in `.env`:
    ADK_RISK_AGENT_CARD_URL=http://127.0.0.1:8080/a2a/risk_agent/.well-known/agent-card.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load env before ADK imports so GCP credentials and project settings are available.
load_dotenv(override=False)

# The ADK LlmAgent instantiates google.genai.Client() lazily; without this flag
# it falls back to Gemini API-key mode and raises `ValueError: No API key was
# provided` at the first model call.  Set it unconditionally so the risk agent
# always uses the Vertex AI backend (which picks up ADC / GOOGLE_APPLICATION_CREDENTIALS).
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    raise SystemExit(
        "GOOGLE_CLOUD_PROJECT is not set. Configure it in .env (or the environment) "
        "before launching the risk-agent server."
    )

# Ensure the project root is the current working directory and is on sys.path,
# so ADK can locate the `risk_agent/` subfolder via `Path.cwd() / agents_dir`.
_PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(_PROJECT_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn  # noqa: E402
from google.adk.cli.fast_api import get_fast_api_app  # noqa: E402

# Folder name (relative to the project root) containing the risk_specialist agent.
_AGENT_FOLDER_NAME = "risk_agent"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bank Risk Specialist Agent — Google ADK A2A Server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ADK 1.31: `agents_dir` is the PARENT folder containing per-agent subfolders.
    # Passing "." (the project root) makes ADK discover `risk_agent/` automatically.
    # `web=False` disables the dev UI; `a2a=True` registers the A2A JSON-RPC
    # endpoint and exposes /a2a/<agent>/.well-known/agent-card.json.
    app = get_fast_api_app(
        agents_dir=".",
        web=False,
        a2a=True,
        host=args.host,
        port=args.port,
    )

    card_url = f"http://{args.host}:{args.port}/a2a/{_AGENT_FOLDER_NAME}/.well-known/agent-card.json"
    rpc_url = f"http://{args.host}:{args.port}/a2a/{_AGENT_FOLDER_NAME}"
    print(
        "\n  Bank Risk Specialist Agent\n"
        f"  A2A server:   http://{args.host}:{args.port}\n"
        f"  JSON-RPC URL: {rpc_url}\n"
        f"  Agent Card:   {card_url}\n"
        f"\n  Set ADK_RISK_AGENT_CARD_URL={card_url}\n"
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
