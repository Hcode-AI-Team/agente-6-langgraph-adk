"""Unit tests for src/config.py — Settings validation and defaults.

Each test that instantiates Settings() must satisfy two conditions:
  1. All required env vars are supplied (via monkeypatch.setenv).
  2. No real .env file is read — achieved by changing the working directory
     to a fresh tmp_path directory that has no .env file.

monkeypatch.setattr(Settings.model_config, "env_file", …) does NOT work
because model_config is a dict subclass: setattr writes to obj.__dict__,
but pydantic-settings reads model_config["env_file"] as a dict key.
monkeypatch.chdir(tmp_path) is the reliable alternative.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings

# Every required field with no default in Settings.
_REQUIRED_VARS = {
    "GOOGLE_CLOUD_PROJECT": "test-project",
    "VECTOR_SEARCH_INDEX_NAME": "projects/123/locations/us-east1/indexes/456",
    "VECTOR_SEARCH_INDEX_ENDPOINT_NAME": (
        "projects/123/locations/us-east1/indexEndpoints/789"
    ),
    "VECTOR_SEARCH_DEPLOYED_INDEX_ID": "deployed_test",
    "ADK_RISK_AGENT_CARD_URL": "https://agent.example.com/.well-known/agent.json",
}


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear lru_cache before and after every test so each test gets a
    fresh Settings() instance unaffected by previous test runs."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def full_env(monkeypatch, tmp_path):
    """Set all required env vars and change CWD to an empty temp directory.

    The CWD change prevents pydantic-settings from finding the project's
    real .env file (which is always resolved relative to CWD).
    """
    for key, value in _REQUIRED_VARS.items():
        monkeypatch.setenv(key, value)
    # Move to a directory that has no .env so pydantic-settings won't load one.
    monkeypatch.chdir(tmp_path)


class TestSettingsDefaults:
    def test_defaults_applied(self, full_env):
        settings = get_settings()
        assert settings.google_cloud_location == "us-central1"
        assert settings.vertex_embedding_model == "text-embedding-004"
        assert settings.vertex_rag_top_k == 5
        assert settings.vertex_llm_model == "gemini-2.5-flash"
        assert settings.vertex_llm_temperature == pytest.approx(0.2)
        assert settings.adk_risk_agent_timeout == 30
        assert settings.log_level == "INFO"

    def test_required_fields_are_read(self, full_env):
        settings = get_settings()
        assert settings.google_cloud_project == "test-project"
        assert settings.vector_search_index_name.startswith("projects/")
        assert settings.vector_search_index_name.endswith("/indexes/456")
        assert settings.vector_search_deployed_index_id == "deployed_test"


class TestSettingsValidation:
    def test_missing_project_raises(self, monkeypatch, tmp_path):
        # Set every required var EXCEPT the project ID, then confirm Settings
        # raises ValidationError listing the missing field.
        for key, value in _REQUIRED_VARS.items():
            if key != "GOOGLE_CLOUD_PROJECT":
                monkeypatch.setenv(key, value)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        # chdir so no .env file can provide the missing variable.
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValidationError) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "google_cloud_project" in str(exc_info.value).lower()

    def test_invalid_log_level_raises(self, full_env, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_log_level_normalised_to_upper(self, full_env, monkeypatch):
        # field_validator must normalise lower-case input to upper-case.
        monkeypatch.setenv("LOG_LEVEL", "debug")
        settings = Settings()  # type: ignore[call-arg]
        assert settings.log_level == "DEBUG"

    def test_temperature_out_of_range_raises(self, full_env, monkeypatch):
        # ge=0.0, le=2.0 — 3.0 must be rejected.
        monkeypatch.setenv("VERTEX_LLM_TEMPERATURE", "3.0")
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_top_k_below_minimum_raises(self, full_env, monkeypatch):
        monkeypatch.setenv("VERTEX_RAG_TOP_K", "0")
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_top_k_above_maximum_raises(self, full_env, monkeypatch):
        monkeypatch.setenv("VERTEX_RAG_TOP_K", "51")
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_invalid_agent_card_url_raises(self, full_env, monkeypatch):
        # HttpUrl field must reject non-URL strings.
        monkeypatch.setenv("ADK_RISK_AGENT_CARD_URL", "not-a-url")
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]


class TestGetSettingsCache:
    def test_returns_same_instance(self, full_env):
        # lru_cache(maxsize=1) must return the identical object on repeated calls.
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_cleared_between_tests(self, full_env):
        # After cache_clear(), a new instance is built — values must still match.
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1.google_cloud_project == s2.google_cloud_project
        # The two instances must be distinct objects (not the cached one).
        assert s1 is not s2

    def test_get_settings_reflects_env_vars(self, full_env, monkeypatch):
        # Changing an env var after cache_clear must produce a new Settings
        # instance that picks up the change.
        monkeypatch.setenv("VERTEX_RAG_TOP_K", "8")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.vertex_rag_top_k == 8
