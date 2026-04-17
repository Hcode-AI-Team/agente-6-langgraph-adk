"""Application settings loaded from environment variables / .env file.

pydantic-settings reads every env var that matches a field name and
validates its type automatically. Any required field (no `default`) that
is missing will raise a `ValidationError` with a clear message before
any external service is contacted.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application configuration.

    Field names match environment variable names (case-insensitive).
    For example, `google_cloud_project` is populated from `GOOGLE_CLOUD_PROJECT`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Unknown env vars are silently ignored — keeps the shell environment clean.
        extra="ignore",
        case_sensitive=False,
    )

    # --- GCP ---
    google_cloud_project: str = Field(..., description="GCP project ID.")
    google_cloud_location: str = Field(
        default="us-central1",
        description="Vertex AI region; must match where the Vector Search index is deployed.",
    )
    # Optional: preferred over Application Default Credentials when set.
    google_application_credentials: Path | None = Field(
        default=None,
        description="Path to a service-account JSON key file.",
    )

    # --- Vertex AI Vector Search ---
    vector_search_index_name: str = Field(
        ...,
        description=(
            "Full resource name of the Vector Search index, e.g. "
            "`projects/<PROJECT_NUMBER>/locations/<REGION>/indexes/<INDEX_ID>`."
        ),
    )
    vector_search_index_endpoint_name: str = Field(
        ...,
        description=(
            "Full resource name of the Index Endpoint, e.g. "
            "`projects/<PROJECT_NUMBER>/locations/<REGION>/indexEndpoints/<ENDPOINT_ID>`."
        ),
    )
    vector_search_deployed_index_id: str = Field(
        ..., description="ID of the deployed index within the endpoint."
    )
    # Optional: GCS bucket used to stage Vector Search datapoints / RAG assets.
    # Not consumed directly by the retrieval path; exposed for future ingestion nodes.
    google_cloud_storage_bucket: str | None = Field(
        default=None,
        description="GCS bucket used to stage Vector Search datapoints / RAG assets.",
    )
    # text-embedding-004 is Vertex AI's recommended dense-retrieval model for this deployment.
    vertex_embedding_model: str = Field(default="text-embedding-004")
    vertex_rag_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of nearest-neighbour documents to retrieve per query.",
    )

    # --- LLM (evaluator and synthesis nodes) ---
    vertex_llm_model: str = Field(default="gemini-2.5-flash")
    # Lower temperature reduces hallucination in policy-compliance contexts.
    vertex_llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- Google ADK — remote risk-agent (A2A) ---
    adk_risk_agent_card_url: HttpUrl = Field(
        ...,
        description="Public URL of the risk-agent's Agent Card (/.well-known/agent.json).",
    )
    adk_risk_agent_timeout: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Seconds to wait for the A2A call before raising a timeout error.",
    )

    # --- Observability ---
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalise to upper-case and reject unknown levels immediately."""
        normalised = value.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalised not in valid:
            raise ValueError(f"Invalid LOG_LEVEL '{value}'. Must be one of {sorted(valid)}.")
        return normalised


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the memoised Settings singleton.

    lru_cache ensures the .env file is read exactly once per process.
    In tests, call `get_settings.cache_clear()` before patching env vars.
    """
    return Settings()  # type: ignore[call-arg]
