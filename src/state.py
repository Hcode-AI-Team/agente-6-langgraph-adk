"""Graph state shared across all nodes of the orchestrator."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class RAGDocument(BaseModel):
    """A single document retrieved from Vertex AI Vector Search.

    Frozen so that nodes cannot accidentally mutate documents already
    stored in the state – all updates must go through the graph merge.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    # Cosine or L2 distance returned by find_neighbors; lower is better.
    distance: float = Field(ge=0.0)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def preview(self, max_chars: int = 240) -> str:
        """Return a truncated, single-line version of the text."""
        cleaned = self.text.strip().replace("\n", " ")
        return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3] + "..."


class ConversationMessage(BaseModel):
    """A single turn stored in the conversation history."""

    model_config = ConfigDict(frozen=True)

    role: str  # "user" | "assistant"
    content: str


class OrchestratorState(BaseModel):
    """Shared state that flows through every node in the LangGraph StateGraph.

    Each node receives the current state and returns a *partial* dict;
    LangGraph merges that dict into the state before calling the next node.
    Fields start with safe defaults so nodes that are skipped (e.g. the
    risk-agent node on simple queries) do not leave the state broken.

    Conversation memory uses two complementary fields:
    - recent_messages: verbatim Q&A pairs from the last N turns (episodic memory).
    - conversation_summary: distilled behavioral patterns extracted from older turns
      (procedural memory). Injected into prompts as soft inference instructions.
    When recent_messages reaches the configured threshold, the summarizer node
    compresses older entries into conversation_summary and prunes the list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The account manager's original question — updated on each invocation.
    question: str

    # --- Conversation memory ---
    # Verbatim Q&A from recent turns. Managed as a plain list so the summarizer
    # can replace it entirely after consolidation.
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    # Accumulated behavioral profile built from consolidated episodic memory.
    # Injected as context into evaluator and synthesis prompts.
    conversation_summary: str = ""

    # Documents retrieved by the RAG node.
    rag_context: list[RAGDocument] = Field(default_factory=list)

    # Set by the evaluator node; drives the conditional edge.
    requires_risk_assessment: bool = False
    evaluator_rationale: str = ""

    # Populated only when requires_risk_assessment is True and the A2A
    # call to the risk-agent node succeeds.
    risk_assessment_response: Optional[str] = None

    # Final answer compiled by the synthesis node.
    final_response: str = ""
