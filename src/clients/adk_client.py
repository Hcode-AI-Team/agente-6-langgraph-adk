"""Google ADK Agent-to-Agent (A2A) client for the remote risk-specialist agent.

The orchestrator acts as the **Agent Client**: it fetches the Agent Card from
the risk-agent server, creates a session, and streams the response.

Flow:
  1. `RemoteA2aAgent` resolves the Agent Card URL to discover the server's
     capabilities and endpoint.
  2. `InMemoryRunner` manages the session lifecycle and drives the event loop.
  3. The `A2APayload` (Pydantic model) is serialised to JSON and placed in the
     `text` field of a `types.Content` message — the A2A standard way to carry
     arbitrary structured data between agents.
  4. We collect only `is_final_response` events and discard intermediate steps.
  5. `asyncio.run` bridges the async ADK runner into the synchronous LangGraph node.
"""

from __future__ import annotations

import asyncio
import uuid
from functools import lru_cache

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from ..config import get_settings
from ..constants import APP_NAME_A2A, USER_ID_A2A
from ..logging_config import get_logger
from ..models import A2APayload

logger = get_logger(__name__)


class A2AClientError(RuntimeError):
    """Raised when the A2A call to the risk-agent fails or times out."""


@lru_cache(maxsize=1)
def _get_remote_agent() -> RemoteA2aAgent:
    """Create and cache the RemoteA2aAgent.

    The constructor fetches the Agent Card from the configured URL, so it
    requires a live network connection.  We cache the result because the
    card is stable for the lifetime of the process.
    """
    settings = get_settings()
    url = str(settings.adk_risk_agent_card_url)
    logger.info("A2A | loading RemoteA2aAgent from %s", url)
    return RemoteA2aAgent(
        name="risk_specialist",
        description="Remote specialist agent for credit-risk analysis at Banco BV.",
        agent_card=url,
    )


@lru_cache(maxsize=1)
def _get_runner() -> InMemoryRunner:
    """Create and cache the InMemoryRunner bound to the remote agent."""
    return InMemoryRunner(agent=_get_remote_agent(), app_name=APP_NAME_A2A)


async def _query_risk_agent_async(payload: A2APayload) -> str:
    """Run the A2A call asynchronously and return the final response text.

    A new session is created for each invocation so that independent
    orchestrator calls do not bleed context into each other.
    """
    settings = get_settings()
    runner = _get_runner()

    # Short random suffix keeps session IDs unique without requiring a DB.
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    await runner.session_service.create_session(
        app_name=APP_NAME_A2A,
        user_id=USER_ID_A2A,
        session_id=session_id,
    )

    # Serialise the Pydantic payload to a JSON string placed in a Content part.
    # This follows the A2A convention of passing structured data via text parts.
    message_text = payload.model_dump_json(indent=2)
    message = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_parts: list[str] = []

    async def _stream() -> None:
        async for event in runner.run_async(
            user_id=USER_ID_A2A,
            session_id=session_id,
            new_message=message,
        ):
            # Only the final response event contains the agent's answer;
            # intermediate events are tool calls, thoughts, and partial outputs.
            if getattr(event, "is_final_response", None) and event.is_final_response():
                content = getattr(event, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            final_parts.append(text)

    try:
        await asyncio.wait_for(_stream(), timeout=settings.adk_risk_agent_timeout)
    except asyncio.TimeoutError as exc:
        raise A2AClientError(
            f"A2A call timed out after {settings.adk_risk_agent_timeout}s "
            "waiting for the risk-agent response."
        ) from exc

    return "\n".join(final_parts).strip() or "[Risk agent returned no assessment]"


def query_risk_agent(payload: A2APayload) -> str:
    """Synchronous wrapper around the async A2A call, for use in LangGraph nodes.

    asyncio.run creates a new event loop each time; this is safe because
    LangGraph nodes run in a regular synchronous context.
    """
    logger.info(
        "A2A | intent=%s chunks=%d trace_id=%s",
        payload.intent,
        len(payload.policy_chunks),
        payload.session.trace_id,
    )
    try:
        response = asyncio.run(_query_risk_agent_async(payload))
    except A2AClientError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in A2A call")
        raise A2AClientError(f"A2A call failed: {exc}") from exc

    logger.info("A2A | response received (%d chars)", len(response))
    return response
