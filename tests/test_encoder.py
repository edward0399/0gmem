"""Tests for Encoder: encoding pipeline for converting text to memories."""

import numpy as np

from zerogmem.encoder.encoder import EncodingResult


class TestEncoder:
    """Tests for the Encoder."""

    def test_encode_returns_encoding_result(self, encoder_with_mock):
        result = encoder_with_mock.encode("Alice loves hiking in the mountains.")
        assert isinstance(result, EncodingResult)
        assert result.memory_item is not None
        assert result.memory_item.content == "Alice loves hiking in the mountains."

    def test_encode_produces_embedding(self, encoder_with_mock):
        result = encoder_with_mock.encode("Hello world")
        emb = result.memory_item.embedding
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (1536,)

    def test_encode_extracts_entities(self, encoder_with_mock):
        result = encoder_with_mock.encode("Alice went to Paris with Bob.")
        # Should extract at least some entities (names, locations)
        assert isinstance(result.entities, list)

    def test_encode_extracts_temporal(self, encoder_with_mock):
        result = encoder_with_mock.encode("I went hiking yesterday morning.")
        assert isinstance(result.temporal_expressions, list)

    def test_encode_importance_score(self, encoder_with_mock):
        result = encoder_with_mock.encode("I got married last summer!")
        importance = result.memory_item.importance
        assert 0.0 <= importance <= 1.0

    def test_encode_detects_negation(self, encoder_with_mock):
        result = encoder_with_mock.encode("I don't like spiders at all.")
        assert isinstance(result.negations, list)

    def test_get_embedding(self, encoder_with_mock):
        emb = encoder_with_mock.get_embedding("test text")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (1536,)

    def test_encode_batch(self, encoder_with_mock):
        texts = ["Hello", "World", "Test"]
        results = encoder_with_mock.encode_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, EncodingResult) for r in results)

    def test_encode_with_speaker(self, encoder_with_mock):
        result = encoder_with_mock.encode("I love hiking", speaker="Alice")
        assert result.memory_item.speaker == "Alice"

    def test_encode_with_session_id(self, encoder_with_mock):
        result = encoder_with_mock.encode("test", session_id="session-42")
        assert result.memory_item.session_id == "session-42"
