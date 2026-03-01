"""Tests for API resilience: retry with exponential backoff in the embedding layer."""

import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zerogmem.encoder.encoder import Encoder, EncoderConfig

# ---------------------------------------------------------------------------
# Fake OpenAI exceptions for testing
# ---------------------------------------------------------------------------


class FakeRateLimitError(Exception):
    pass


class FakeAPIConnectionError(Exception):
    pass


class FakeAPITimeoutError(Exception):
    pass


class FakeInternalServerError(Exception):
    pass


class FakeAuthenticationError(Exception):
    pass


class FakeBadRequestError(Exception):
    pass


def _mock_openai_module():
    """Create a mock openai module with exception classes."""
    mod = MagicMock()
    mod.RateLimitError = FakeRateLimitError
    mod.APIConnectionError = FakeAPIConnectionError
    mod.APITimeoutError = FakeAPITimeoutError
    mod.InternalServerError = FakeInternalServerError
    mod.AuthenticationError = FakeAuthenticationError
    mod.BadRequestError = FakeBadRequestError
    mod.NotFoundError = LookupError
    mod.PermissionDeniedError = PermissionError
    return mod


def _mock_embedding_response():
    """Create a mock OpenAI embedding response."""
    response = MagicMock()
    response.data = [MagicMock()]
    response.data[0].embedding = list(np.zeros(1536, dtype=np.float32))
    return response


def _make_encoder_with_mock(mock_module, mock_client, max_retries=3):
    """Create an Encoder that uses a mock openai module."""
    mock_module.OpenAI.return_value = mock_client
    with patch.dict(sys.modules, {"openai": mock_module}):
        encoder = Encoder(config=EncoderConfig(max_retries=max_retries))
        # Force creation of the embed function with our mocked openai
        embed_fn = encoder._get_embedding_fn()
    return embed_fn


# ---------------------------------------------------------------------------
# Encoder retry tests
# ---------------------------------------------------------------------------


class TestEncoderRetry:
    """Tests for retry behavior in Encoder._get_embedding_fn."""

    def test_succeeds_on_first_try(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _mock_embedding_response()

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client)
        result = embed_fn("hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1536,)
        assert mock_client.embeddings.create.call_count == 1

    def test_retries_on_rate_limit_then_succeeds(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            FakeRateLimitError("rate limited"),
            FakeRateLimitError("rate limited"),
            _mock_embedding_response(),
        ]

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client)
        result = embed_fn("hello world")

        assert isinstance(result, np.ndarray)
        assert mock_client.embeddings.create.call_count == 3

    def test_retries_on_connection_error(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            FakeAPIConnectionError("connection reset"),
            _mock_embedding_response(),
        ]

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client)
        result = embed_fn("hello")

        assert isinstance(result, np.ndarray)
        assert mock_client.embeddings.create.call_count == 2

    def test_retries_on_internal_server_error(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            FakeInternalServerError("500"),
            _mock_embedding_response(),
        ]

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client)
        result = embed_fn("hello")

        assert isinstance(result, np.ndarray)
        assert mock_client.embeddings.create.call_count == 2

    def test_exhausts_retries_raises(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = FakeRateLimitError("rate limited")

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client, max_retries=2)

        with pytest.raises(FakeRateLimitError):
            embed_fn("hello world")

        assert mock_client.embeddings.create.call_count == 2

    def test_no_retry_on_auth_error(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = FakeAuthenticationError("bad key")

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client, max_retries=3)

        with pytest.raises(FakeAuthenticationError):
            embed_fn("hello world")

        # Should have called exactly once (no retries)
        assert mock_client.embeddings.create.call_count == 1

    def test_no_retry_on_bad_request(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = FakeBadRequestError("invalid input")

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client, max_retries=3)

        with pytest.raises(FakeBadRequestError):
            embed_fn("hello world")

        assert mock_client.embeddings.create.call_count == 1

    def test_max_retries_configurable(self):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = FakeRateLimitError("rate limited")

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client, max_retries=5)

        with pytest.raises(FakeRateLimitError):
            embed_fn("hello")

        assert mock_client.embeddings.create.call_count == 5

    def test_fallback_to_random_when_no_openai(self):
        with patch.dict(sys.modules, {"openai": None}):
            encoder = Encoder(config=EncoderConfig())
            embed_fn = encoder._get_embedding_fn()
            result = embed_fn("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1536,)

    def test_retry_logs_warning(self, caplog):
        mock_mod = _mock_openai_module()
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            FakeRateLimitError("rate limited"),
            _mock_embedding_response(),
        ]

        embed_fn = _make_encoder_with_mock(mock_mod, mock_client, max_retries=3)

        with caplog.at_level(logging.WARNING, logger="0gmem.encoder"):
            result = embed_fn("hello")

        assert isinstance(result, np.ndarray)
        # tenacity's before_sleep_log produces a log entry
        assert any("Retrying" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestEncoderConfigMaxRetries:

    def test_default_value(self):
        config = EncoderConfig()
        assert config.max_retries == 3

    def test_custom_value(self):
        config = EncoderConfig(max_retries=10)
        assert config.max_retries == 10


# ---------------------------------------------------------------------------
# MCP server env var tests
# ---------------------------------------------------------------------------


class TestMcpServerRetryConfig:

    def test_default_max_retries(self, monkeypatch):
        monkeypatch.delenv("ZEROGMEM_API_MAX_RETRIES", raising=False)
        monkeypatch.delenv("ZEROGMEM_EMBEDDING_MODEL", raising=False)
        from zerogmem.mcp_server import _build_configs

        encoder_config, _, _ = _build_configs()
        assert encoder_config.max_retries == 3

    def test_custom_max_retries(self, monkeypatch):
        monkeypatch.setenv("ZEROGMEM_API_MAX_RETRIES", "7")
        monkeypatch.delenv("ZEROGMEM_EMBEDDING_MODEL", raising=False)
        from zerogmem.mcp_server import _build_configs

        encoder_config, _, _ = _build_configs()
        assert encoder_config.max_retries == 7

    def test_invalid_max_retries_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("ZEROGMEM_API_MAX_RETRIES", "not_a_number")
        monkeypatch.delenv("ZEROGMEM_EMBEDDING_MODEL", raising=False)
        from zerogmem.mcp_server import _build_configs

        with caplog.at_level(logging.WARNING, logger="0gmem-mcp"):
            encoder_config, _, _ = _build_configs()
        assert encoder_config.max_retries == 3
