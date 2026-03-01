"""Tests for WorkingMemory: attention-based memory with limited capacity."""

import numpy as np
from datetime import datetime, timedelta

from zerogmem.memory.working import WorkingMemory, WorkingMemoryItem


class TestWorkingMemoryItem:
    """Tests for the WorkingMemoryItem dataclass."""

    def test_boost_increases_weight(self):
        item = WorkingMemoryItem(id="a", content="test", attention_weight=0.5)
        item.boost(0.3)
        assert item.attention_weight == 0.8
        assert item.access_count == 1

    def test_boost_capped_at_one(self):
        item = WorkingMemoryItem(id="a", content="test", attention_weight=0.9)
        item.boost(0.5)
        assert item.attention_weight == 1.0

    def test_decay_reduces_weight(self):
        # Set last_accessed to 10 minutes ago so decay has measurable effect
        past = datetime.now() - timedelta(minutes=10)
        item = WorkingMemoryItem(
            id="a", content="test", attention_weight=1.0, last_accessed=past
        )
        item.decay(rate=0.1, time_factor=1.0)
        assert item.attention_weight < 1.0

    def test_decay_floors_at_001(self):
        past = datetime.now() - timedelta(hours=1)
        item = WorkingMemoryItem(
            id="a", content="test", attention_weight=0.02, last_accessed=past
        )
        item.decay(rate=1.0, time_factor=1.0)
        assert item.attention_weight == 0.01


class TestWorkingMemoryBasic:
    """Tests for basic WorkingMemory operations."""

    def test_add_item(self, working_memory, make_wm_item):
        item = make_wm_item("hello world")
        result = working_memory.add(item)
        assert result is True
        assert working_memory.size == 1

    def test_add_duplicate_boosts_existing(self, working_memory, make_wm_item):
        item = make_wm_item("hello", item_id="dup", attention=0.5)
        working_memory.add(item)
        initial_count = item.access_count

        # Add again with same id -> should boost
        item2 = make_wm_item("hello", item_id="dup", attention=0.5)
        working_memory.add(item2)
        assert working_memory.size == 1
        assert item.access_count > initial_count

    def test_get_boosts_attention(self, working_memory, make_wm_item):
        item = make_wm_item("test", item_id="x", attention=0.5)
        working_memory.add(item)
        retrieved = working_memory.get("x")
        assert retrieved is not None
        assert retrieved.attention_weight > 0.5

    def test_get_nonexistent_returns_none(self, working_memory):
        assert working_memory.get("nonexistent") is None

    def test_remove_item(self, working_memory, make_wm_item):
        item = make_wm_item("test", item_id="r")
        working_memory.add(item)
        assert working_memory.remove("r") is True
        assert working_memory.size == 0

    def test_remove_nonexistent_returns_false(self, working_memory):
        assert working_memory.remove("nope") is False

    def test_clear(self, working_memory, make_wm_item):
        for i in range(3):
            working_memory.add(make_wm_item(f"item {i}"))
        working_memory.clear()
        assert working_memory.size == 0

    def test_size_property(self, working_memory, make_wm_item):
        assert working_memory.size == 0
        working_memory.add(make_wm_item("a"))
        working_memory.add(make_wm_item("b"))
        assert working_memory.size == 2

    def test_is_full_property(self, working_memory, make_wm_item):
        assert not working_memory.is_full
        for i in range(5):
            working_memory.add(make_wm_item(f"item {i}"))
        assert working_memory.is_full


class TestWorkingMemoryCapacity:
    """Tests for capacity management and eviction."""

    def test_at_capacity_evicts_lowest(self, working_memory, make_wm_item):
        # Fill to capacity=5, item "low" has lowest attention
        working_memory.add(make_wm_item("low", item_id="low", attention=0.05))
        for i in range(4):
            working_memory.add(make_wm_item(f"high {i}", attention=0.9))
        assert working_memory.size == 5

        # Adding one more should evict "low"
        working_memory.add(make_wm_item("overflow", attention=0.9))
        assert working_memory.size == 5
        assert working_memory.get("low") is None  # Note: get() would boost, but returns None

    def test_force_add_at_capacity(self, working_memory, make_wm_item):
        for i in range(5):
            working_memory.add(make_wm_item(f"item {i}", attention=0.9))
        # Force add even at capacity
        result = working_memory.add(make_wm_item("forced"), force=True)
        assert result is True


class TestWorkingMemoryAttention:
    """Tests for attention and context retrieval."""

    def test_update_attention_positive(self, working_memory, make_wm_item):
        item = make_wm_item("test", item_id="u", attention=0.5)
        working_memory.add(item)
        working_memory.update_attention("u", 0.3)
        assert item.attention_weight == 0.8

    def test_update_attention_negative(self, working_memory, make_wm_item):
        item = make_wm_item("test", item_id="u", attention=0.5)
        working_memory.add(item)
        working_memory.update_attention("u", -0.4)
        assert abs(item.attention_weight - 0.1) < 1e-9

    def test_update_attention_clamped(self, working_memory, make_wm_item):
        item = make_wm_item("test", item_id="u", attention=0.5)
        working_memory.add(item)
        working_memory.update_attention("u", -1.0)
        assert item.attention_weight == 0.01
        working_memory.update_attention("u", 2.0)
        assert item.attention_weight == 1.0

    def test_get_all_min_attention(self, working_memory, make_wm_item):
        working_memory.add(make_wm_item("low", attention=0.1))
        working_memory.add(make_wm_item("high", attention=0.9))
        results = working_memory.get_all(min_attention=0.5)
        assert len(results) == 1
        assert results[0].content == "high"

    def test_get_context_no_query(self, working_memory, make_wm_item):
        working_memory.add(make_wm_item("a", attention=0.3))
        working_memory.add(make_wm_item("b", attention=0.9))
        ctx = working_memory.get_context(top_k=2)
        assert len(ctx) == 2
        # First should be highest attention
        assert ctx[0].content == "b"

    def test_get_context_with_query(self, working_memory, make_wm_item, mock_embedding_fn):
        working_memory.add(make_wm_item("hiking in mountains", attention=0.3))
        working_memory.add(make_wm_item("cooking dinner", attention=0.9))
        query_emb = mock_embedding_fn("hiking in mountains")
        ctx = working_memory.get_context(query_embedding=query_emb, top_k=2)
        assert len(ctx) == 2

    def test_get_context_empty(self, working_memory):
        ctx = working_memory.get_context()
        assert ctx == []

    def test_get_stats(self, working_memory, make_wm_item):
        stats = working_memory.get_stats()
        assert stats["size"] == 0
        assert stats["avg_attention"] == 0.0

        working_memory.add(make_wm_item("test", attention=0.6))
        stats = working_memory.get_stats()
        assert stats["size"] == 1
        assert stats["capacity"] == 5

    def test_get_summary_empty(self, working_memory):
        assert "empty" in working_memory.get_summary().lower()

    def test_to_context_string(self, working_memory, make_wm_item):
        working_memory.add(make_wm_item("hello world"))
        ctx = working_memory.to_context_string()
        assert "hello world" in ctx
