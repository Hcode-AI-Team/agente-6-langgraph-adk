"""Shared Gemini chat client used by the evaluator and synthesis nodes.

Uses `langchain-google-genai` with the Vertex AI backend (instead of the
deprecated `langchain-google-vertexai.ChatVertexAI`).  The Vertex AI backend
is selected via the `GOOGLE_GENAI_USE_VERTEXAI` environment variable, which we
set programmatically from `Settings` so no extra env configuration is needed.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return a memoised ChatGoogleGenerativeAI instance.

    Constructing the chat client initialises an HTTP client; caching it avoids
    that overhead on repeated invocations within the same process.
    """
    settings = get_settings()

    # Switch the google-genai SDK to Vertex AI backend *before* the chat client
    # reads these values at construction time.
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.google_cloud_project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.google_cloud_location)

    logger.info(
        "Initialising ChatGoogleGenerativeAI (Vertex AI backend): model=%s location=%s temperature=%.2f",
        settings.vertex_llm_model,
        settings.google_cloud_location,
        settings.vertex_llm_temperature,
    )
    return ChatGoogleGenerativeAI(
        model=settings.vertex_llm_model,
        temperature=settings.vertex_llm_temperature,
    )
