"""Tests for environment variable configuration of the MCP server."""

import logging

import pytest


# ---------------------------------------------------------------------------
# _env_int helper
# ---------------------------------------------------------------------------

class TestEnvInt:
    """Tests for the _env_int helper function."""

    def test_returns_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("ZEROGMEM_TEST_VAR", raising=False)
        from zerogmem.mcp_server import _env_int
        assert _env_int("ZEROGMEM_TEST_VAR", 42) == 42

    def test_reads_valid_integer(self, monkeypatch):
        monkeypatch.setenv("ZEROGMEM_TEST_VAR", "100")
        from zerogmem.mcp_server import _env_int
        assert _env_int("ZEROGMEM_TEST_VAR", 42) == 100

    def test_invalid_returns_default_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("ZEROGMEM_TEST_VAR", "not_a_number")
        from zerogmem.mcp_server import _env_int
        with caplog.at_level(logging.WARNING, logger="0gmem-mcp"):
            result = _env_int("ZEROGMEM_TEST_VAR", 42)
        assert result == 42
        assert "Invalid integer" in caplog.text

    def test_reads_zero(self, monkeypatch):
        monkeypatch.setenv("ZEROGMEM_TEST_VAR", "0")
        from zerogmem.mcp_server import _env_int
        assert _env_int("ZEROGMEM_TEST_VAR", 42) == 0

    def test_reads_negative(self, monkeypatch):
        monkeypatch.setenv("ZEROGMEM_TEST_VAR", "-5")
        from zerogmem.mcp_server import _env_int
        assert _env_int("ZEROGMEM_TEST_VAR", 42) == -5


# ---------------------------------------------------------------------------
# _build_configs
# ---------------------------------------------------------------------------

class TestBuildConfigs:
    """Tests for the _build_configs factory function."""

    def test_defaults_without_env_vars(self, monkeypatch):
        # Clear all relevant env vars
        for var in [
            "ZEROGMEM_EMBEDDING_MODEL",
            "ZEROGMEM_MAX_EPISODES",
            "ZEROGMEM_MAX_FACTS",
            "ZEROGMEM_WORKING_MEMORY_CAPACITY",
            "ZEROGMEM_MAX_CONTEXT_TOKENS",
        ]:
            monkeypatch.delenv(var, raising=False)

        from zerogmem.mcp_server import _build_configs
        encoder_config, memory_config, retriever_config = _build_configs()

        assert encoder_config.embedding_model == "text-embedding-3-small"
        assert memory_config.max_episodes == 500
        assert memory_config.max_facts == 5000
        assert memory_config.working_memory_capacity == 20
        assert retriever_config.max_context_tokens == 8000

    def test_custom_values_from_env(self, monkeypatch):
        monkeypatch.setenv("ZEROGMEM_EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("ZEROGMEM_MAX_EPISODES", "100")
        monkeypatch.setenv("ZEROGMEM_MAX_FACTS", "1000")
        monkeypatch.setenv("ZEROGMEM_WORKING_MEMORY_CAPACITY", "50")
        monkeypatch.setenv("ZEROGMEM_MAX_CONTEXT_TOKENS", "16000")

        from zerogmem.mcp_server import _build_configs
        encoder_config, memory_config, retriever_config = _build_configs()

        assert encoder_config.embedding_model == "text-embedding-3-large"
        assert memory_config.max_episodes == 100
        assert memory_config.max_facts == 1000
        assert memory_config.working_memory_capacity == 50
        assert retriever_config.max_context_tokens == 16000

    def test_partial_env_vars(self, monkeypatch):
        # Only set some vars; others should use defaults
        for var in [
            "ZEROGMEM_EMBEDDING_MODEL",
            "ZEROGMEM_MAX_EPISODES",
            "ZEROGMEM_MAX_FACTS",
            "ZEROGMEM_WORKING_MEMORY_CAPACITY",
            "ZEROGMEM_MAX_CONTEXT_TOKENS",
        ]:
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("ZEROGMEM_MAX_EPISODES", "250")

        from zerogmem.mcp_server import _build_configs
        encoder_config, memory_config, retriever_config = _build_configs()

        assert memory_config.max_episodes == 250  # custom
        assert memory_config.max_facts == 5000    # default
        assert encoder_config.embedding_model == "text-embedding-3-small"  # default
        assert retriever_config.max_context_tokens == 8000  # default

    def test_invalid_int_falls_back_to_default(self, monkeypatch, caplog):
        for var in [
            "ZEROGMEM_EMBEDDING_MODEL",
            "ZEROGMEM_MAX_EPISODES",
            "ZEROGMEM_MAX_FACTS",
            "ZEROGMEM_WORKING_MEMORY_CAPACITY",
            "ZEROGMEM_MAX_CONTEXT_TOKENS",
        ]:
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("ZEROGMEM_MAX_EPISODES", "abc")

        from zerogmem.mcp_server import _build_configs
        with caplog.at_level(logging.WARNING, logger="0gmem-mcp"):
            _, memory_config, _ = _build_configs()

        assert memory_config.max_episodes == 500  # fell back to default
        assert "Invalid integer" in caplog.text
