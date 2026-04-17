"""Node 1 — RAG retrieval: query Vertex AI Vector Search for relevant policy documents."""

from __future__ import annotations

from ..clients.vector_search import search_policies
from ..logging_config import get_logger
from ..state import OrchestratorState

logger = get_logger(__name__)


def node_rag_retrieval(state: OrchestratorState) -> dict:
    """Embed the question and fetch the nearest policy chunks from Vector Search.

    Returns a partial state dict with `rag_context` populated.  An empty list
    is valid — the evaluator node handles the case where no documents are found.
    """
    logger.info("node_rag_retrieval | question=%r", state.question)
    documents = search_policies(state.question)
    return {"rag_context": documents}
