"""Vertex AI Vector Search client — embedding generation and nearest-neighbour retrieval.

Responsibilities:
1. Embed a query string using the configured Vertex AI text-embedding model.
2. Call `find_neighbors` on the deployed MatchingEngineIndexEndpoint.
3. Deserialise the raw response into typed `RAGDocument` objects.

Both the embedding model and the index endpoint are cached at module level
because they are stateless SDK wrappers: there is no benefit to reinstantiating
them on every query, and the SDK initialisation involves a network round-trip.
"""

from __future__ import annotations

from functools import lru_cache

from google import genai
from google.cloud import aiplatform
from google.genai import types as genai_types

from ..config import get_settings
from ..logging_config import get_logger
from ..state import RAGDocument

logger = get_logger(__name__)

# The Vertex AI restrict namespace that stores the policy-text token.
# This value must match the namespace used when populating the index datapoints.
_TEXT_NAMESPACE = "texto"


class VectorSearchError(RuntimeError):
    """Raised when the Vector Search call fails or returns an unexpected response."""


@lru_cache(maxsize=1)
def _init_aiplatform() -> None:
    """Initialise the aiplatform SDK once per process.

    Subsequent calls within the same process are no-ops because of lru_cache.
    The SDK uses Application Default Credentials (or the key file from
    GOOGLE_APPLICATION_CREDENTIALS) unless credentials are passed explicitly.
    """
    settings = get_settings()
    aiplatform.init(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )
    logger.debug(
        "aiplatform.init: project=%s, location=%s",
        settings.google_cloud_project,
        settings.google_cloud_location,
    )


@lru_cache(maxsize=1)
def _get_genai_client() -> genai.Client:
    """Return a cached google-genai client bound to the Vertex AI backend.

    This replaces the deprecated `vertexai.language_models.TextEmbeddingModel`
    (google-cloud-aiplatform SDK) with the unified `google-genai` SDK.
    """
    settings = get_settings()
    logger.info(
        "Initialising google-genai client (Vertex AI backend) for embedding model: %s",
        settings.vertex_embedding_model,
    )
    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


@lru_cache(maxsize=1)
def _get_index_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """Load and cache the MatchingEngineIndexEndpoint.

    Accepts either a numeric endpoint ID or the full resource name
    (`projects/.../indexEndpoints/<id>`); we store the full resource name
    in the settings for explicitness.
    """
    _init_aiplatform()
    settings = get_settings()
    logger.info(
        "Loading index endpoint %s (deployed_id=%s)",
        settings.vector_search_index_endpoint_name,
        settings.vector_search_deployed_index_id,
    )
    return aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=settings.vector_search_index_endpoint_name
    )


def embed_query(text: str) -> list[float]:
    """Convert a query string into a dense vector using RETRIEVAL_QUERY task type.

    RETRIEVAL_QUERY produces asymmetric embeddings optimised for query-to-document
    similarity, rather than symmetric document-to-document similarity.
    """
    if not text.strip():
        raise VectorSearchError("Cannot embed an empty query string.")

    settings = get_settings()
    client = _get_genai_client()
    response = client.models.embed_content(
        model=settings.vertex_embedding_model,
        contents=[text],
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    if not response.embeddings or response.embeddings[0].values is None:
        raise VectorSearchError("Embedding response did not contain any vectors.")
    return list(response.embeddings[0].values)


def _extract_text_and_metadata(neighbor: object) -> tuple[str, dict]:
    """Pull policy text and metadata from a MatchNeighbor response object.

    Convention: the Vector Search index was populated with datapoints that store
    the chunk text as an `allow_tokens` value inside a restrict with
    `namespace=_TEXT_NAMESPACE`.  All other restricts are collected as metadata.
    If the index uses a different storage scheme (e.g. a Firestore metadata store),
    replace this function's body while keeping its signature.
    """
    metadata: dict = {}
    text = ""

    restricts = getattr(neighbor, "restricts", None) or []
    for restrict in restricts:
        ns = getattr(restrict, "namespace", None)
        tokens = list(getattr(restrict, "allow_tokens", []) or [])
        if ns:
            metadata[ns] = tokens
            if ns == _TEXT_NAMESPACE and tokens:
                text = tokens[0]

    crowding_tag = getattr(neighbor, "crowding_tag", None)
    if crowding_tag:
        metadata["crowding_tag"] = crowding_tag

    return text, metadata


def search_policies(question: str, top_k: int | None = None) -> list[RAGDocument]:
    """Retrieve the top-k policy documents most semantically similar to `question`.

    Returns an empty list when the index returns no neighbours, rather than
    raising, so the evaluator node can decide how to handle low coverage.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.vertex_rag_top_k

    logger.info("Vector Search | querying (top_k=%d)", k)
    embedding = embed_query(question)

    try:
        endpoint = _get_index_endpoint()
        # find_neighbors accepts a list of query vectors; we always send one.
        matches = endpoint.find_neighbors(
            deployed_index_id=settings.vector_search_deployed_index_id,
            queries=[embedding],
            num_neighbors=k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("find_neighbors call failed")
        raise VectorSearchError(f"Vector Search query failed: {exc}") from exc

    if not matches or not matches[0]:
        logger.warning("Vector Search | no neighbours returned")
        return []

    documents: list[RAGDocument] = []
    for neighbor in matches[0]:
        text, metadata = _extract_text_and_metadata(neighbor)
        documents.append(
            RAGDocument(
                id=str(getattr(neighbor, "id", "")),
                distance=float(getattr(neighbor, "distance", 0.0)),
                # Fallback message when the text namespace is empty — signals that
                # the caller should fetch the full text from the metadata store.
                text=text or f"[Document {neighbor.id}] (fetch from metadata store)",
                metadata=metadata,
            )
        )

    logger.info("Vector Search | retrieved %d documents", len(documents))
    return documents
