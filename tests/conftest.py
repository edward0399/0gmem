"""
Shared pytest fixtures for 0GMem tests.
"""

import uuid

import pytest
import numpy as np
from typing import Callable
from datetime import datetime, timedelta

from zerogmem import MemoryManager, Encoder, Retriever, MemoryConfig
from zerogmem.encoder.encoder import EncoderConfig
from zerogmem.memory.working import WorkingMemory, WorkingMemoryItem
from zerogmem.memory.episodic import EpisodicMemory, Episode, EpisodeMessage
from zerogmem.memory.semantic import SemanticMemoryStore, Fact
from zerogmem.graph.entity import EntityGraph, EntityNode, EntityEdge, EntityType
from zerogmem.graph.temporal import TemporalGraph, TemporalNode, TimeInterval
from zerogmem.graph.semantic import SemanticGraph, SemanticNode, SemanticEdge
from zerogmem.graph.unified import UnifiedMemoryGraph, UnifiedMemoryItem
from zerogmem.retriever.attention_filter import AttentionFilter, FilterConfig


# ---------------------------------------------------------------------------
# Core fixtures (existing)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_fn() -> Callable[[str], np.ndarray]:
    """Create a mock embedding function for testing without API calls."""
    def embed(text: str) -> np.ndarray:
        # Create deterministic embeddings based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(1536).astype(np.float32)
    return embed


@pytest.fixture
def memory_config() -> MemoryConfig:
    """Create a memory configuration for testing."""
    return MemoryConfig(
        working_memory_capacity=10,
        working_memory_decay_rate=0.1,
        embedding_dim=1536,
    )


@pytest.fixture
def memory_manager(memory_config: MemoryConfig) -> MemoryManager:
    """Create a MemoryManager instance for testing."""
    return MemoryManager(config=memory_config)


@pytest.fixture
def memory_with_embeddings(memory_manager: MemoryManager, mock_embedding_fn) -> MemoryManager:
    """Create a MemoryManager with mock embedding function."""
    memory_manager.set_embedding_function(mock_embedding_fn)
    return memory_manager


@pytest.fixture
def populated_memory(memory_with_embeddings: MemoryManager) -> MemoryManager:
    """Create a MemoryManager populated with test data."""
    memory = memory_with_embeddings

    # Add a test conversation
    memory.start_session()
    memory.add_message("Alice", "I love hiking in the mountains.")
    memory.add_message("Bob", "Which mountains have you visited?")
    memory.add_message("Alice", "I went to the Alps last summer. The Matterhorn was incredible!")
    memory.add_message("Bob", "That sounds amazing!")
    memory.end_session()

    return memory


@pytest.fixture
def retriever(populated_memory: MemoryManager, mock_embedding_fn) -> Retriever:
    """Create a Retriever instance with populated memory."""
    return Retriever(populated_memory, embedding_fn=mock_embedding_fn)


# ---------------------------------------------------------------------------
# Working Memory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def working_memory():
    """Small-capacity working memory for testing."""
    return WorkingMemory(capacity=5, decay_rate=0.05, eviction_threshold=0.1)


@pytest.fixture
def make_wm_item(mock_embedding_fn):
    """Factory to create WorkingMemoryItem instances."""
    def _make(content, item_id=None, attention=1.0):
        return WorkingMemoryItem(
            id=item_id or str(uuid.uuid4()),
            content=content,
            embedding=mock_embedding_fn(content),
            attention_weight=attention,
        )
    return _make


# ---------------------------------------------------------------------------
# Episodic Memory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def episodic_memory():
    """Empty episodic memory store."""
    return EpisodicMemory()


@pytest.fixture
def make_episode(mock_embedding_fn):
    """Factory to create Episode instances."""
    def _make(
        participants=None,
        topics=None,
        messages=None,
        session_id=None,
        start_time=None,
    ):
        ep = Episode(
            session_id=session_id or str(uuid.uuid4()),
            participants=participants or [],
            participant_names=participants or [],
            topics=topics or [],
            start_time=start_time or datetime(2024, 6, 15, 12, 0, 0),
            summary_embedding=mock_embedding_fn("episode summary"),
        )
        for speaker, content in (messages or []):
            ep.add_message(EpisodeMessage(speaker=speaker, content=content))
        return ep
    return _make


# ---------------------------------------------------------------------------
# Semantic Memory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def semantic_store():
    """Empty semantic memory store."""
    return SemanticMemoryStore()


@pytest.fixture
def make_fact(mock_embedding_fn):
    """Factory to create Fact instances."""
    def _make(subject, predicate, obj, category="", negated=False, confidence=1.0):
        return Fact(
            content=f"{subject} {predicate} {obj}",
            subject=subject,
            predicate=predicate,
            object=obj,
            category=category,
            negated=negated,
            confidence=confidence,
            embedding=mock_embedding_fn(f"{subject} {predicate} {obj}"),
            sources=["test-source"],
        )
    return _make


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def entity_graph():
    """Empty entity graph."""
    return EntityGraph()


@pytest.fixture
def temporal_graph():
    """Empty temporal graph."""
    return TemporalGraph()


@pytest.fixture
def semantic_graph():
    """Empty semantic graph."""
    return SemanticGraph(embedding_dim=1536)


@pytest.fixture
def unified_graph():
    """Empty unified memory graph."""
    return UnifiedMemoryGraph(embedding_dim=1536)


# ---------------------------------------------------------------------------
# Encoder / Retriever component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder_with_mock(mock_embedding_fn):
    """Encoder using mock embeddings (no API calls)."""
    return Encoder(config=EncoderConfig(), embedding_fn=mock_embedding_fn)


@pytest.fixture
def attention_filter(mock_embedding_fn):
    """Attention filter with small token budget for testing."""
    config = FilterConfig(
        relevance_threshold=0.3,
        max_context_tokens=500,
        diversity_weight=0.3,
        semantic_similarity_threshold=0.85,
    )
    return AttentionFilter(config=config, embedding_fn=mock_embedding_fn)
