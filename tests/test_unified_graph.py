"""Tests for UnifiedMemoryGraph: cross-graph coordination."""

from datetime import datetime

import numpy as np

from zerogmem.graph.unified import UnifiedMemoryGraph, UnifiedMemoryItem
from zerogmem.graph.entity import EntityNode, EntityType
from zerogmem.graph.temporal import TimeInterval


class TestUnifiedMemoryGraph:
    """Tests for the UnifiedMemoryGraph."""

    def _make_memory(self, content, embedding_fn, event_time=None, entities=None, concepts=None):
        return UnifiedMemoryItem(
            content=content,
            embedding=embedding_fn(content),
            event_time=event_time,
            entities=entities or [],
            concepts=concepts or [],
        )

    def test_add_memory_creates_semantic_node(self, unified_graph, mock_embedding_fn):
        mem = self._make_memory("hiking trip", mock_embedding_fn)
        mid = unified_graph.add_memory(mem)
        assert mid == mem.id
        assert mem.semantic_node_id is not None
        assert len(unified_graph.semantic_graph.nodes) == 1

    def test_add_memory_creates_temporal_node(self, unified_graph, mock_embedding_fn):
        mem = self._make_memory(
            "lunch meeting",
            mock_embedding_fn,
            event_time=TimeInterval(start=datetime(2024, 6, 1, 12, 0)),
        )
        unified_graph.add_memory(mem)
        assert mem.temporal_node_id is not None
        assert len(unified_graph.temporal_graph.nodes) == 1

    def test_add_entity(self, unified_graph):
        alice = EntityNode(name="Alice", entity_type=EntityType.PERSON)
        eid = unified_graph.add_entity(alice)
        assert eid == alice.id
        assert unified_graph.get_entity(eid) is not None

    def test_add_entity_relation(self, unified_graph):
        alice = EntityNode(name="Alice", entity_type=EntityType.PERSON)
        bob = EntityNode(name="Bob", entity_type=EntityType.PERSON)
        unified_graph.add_entity(alice)
        unified_graph.add_entity(bob)
        eid = unified_graph.add_entity_relation(alice.id, bob.id, "knows")
        assert eid is not None

    def test_query_by_time(self, unified_graph, mock_embedding_fn):
        mem = self._make_memory(
            "morning jog",
            mock_embedding_fn,
            event_time=TimeInterval(
                start=datetime(2024, 6, 1, 8, 0),
                end=datetime(2024, 6, 1, 9, 0),
            ),
        )
        unified_graph.add_memory(mem)

        results = unified_graph.query_by_time(
            start=datetime(2024, 6, 1, 0, 0),
            end=datetime(2024, 6, 1, 23, 59),
        )
        assert len(results) == 1
        assert results[0].content == "morning jog"

    def test_query_by_similarity(self, unified_graph, mock_embedding_fn):
        unified_graph.add_memory(self._make_memory("hiking trip", mock_embedding_fn))
        unified_graph.add_memory(self._make_memory("cooking class", mock_embedding_fn))

        query_emb = mock_embedding_fn("hiking trip")
        results = unified_graph.query_by_similarity(query_emb, top_k=5)
        assert len(results) >= 1
        # First result should be the most similar
        assert results[0][0].content == "hiking trip"

    def test_query_by_entity(self, unified_graph, mock_embedding_fn):
        alice = EntityNode(name="Alice", entity_type=EntityType.PERSON)
        unified_graph.add_entity(alice)

        mem = self._make_memory("hiking trip", mock_embedding_fn, entities=[alice.id])
        unified_graph.add_memory(mem)

        results = unified_graph.query_by_entity(alice.id)
        assert len(results) == 1

    def test_query_by_concept(self, unified_graph, mock_embedding_fn):
        mem = self._make_memory("hiking trip", mock_embedding_fn, concepts=["hiking", "outdoors"])
        unified_graph.add_memory(mem)

        results = unified_graph.query_by_concept("hiking")
        assert len(results) == 1
        assert unified_graph.query_by_concept("nonexistent") == []

    def test_add_and_check_negative_fact(self, unified_graph):
        alice = EntityNode(name="Alice", entity_type=EntityType.PERSON)
        bob = EntityNode(name="Bob", entity_type=EntityType.PERSON)
        unified_graph.add_entity(alice)
        unified_graph.add_entity(bob)

        unified_graph.add_negative_fact(alice.id, "likes", bob.id, "ep-1")

        is_negated, evidence = unified_graph.check_negation(alice.id, "likes", bob.id)
        assert is_negated is True
        assert evidence == "ep-1"

    def test_check_negation_not_found(self, unified_graph):
        alice = EntityNode(name="Alice", entity_type=EntityType.PERSON)
        bob = EntityNode(name="Bob", entity_type=EntityType.PERSON)
        unified_graph.add_entity(alice)
        unified_graph.add_entity(bob)

        is_negated, evidence = unified_graph.check_negation(alice.id, "likes", bob.id)
        assert is_negated is False
        assert evidence is None

    def test_get_stats(self, unified_graph, mock_embedding_fn):
        stats = unified_graph.get_stats()
        assert stats["total_memories"] == 0

        unified_graph.add_memory(self._make_memory(
            "event", mock_embedding_fn,
            event_time=TimeInterval(start=datetime(2024, 6, 1)),
            concepts=["test"],
        ))
        stats = unified_graph.get_stats()
        assert stats["total_memories"] == 1
        assert stats["temporal_nodes"] == 1
        assert stats["semantic_nodes"] == 1
        assert stats["unique_concepts"] == 1

    def test_multi_hop_query(self, unified_graph, mock_embedding_fn):
        alice = EntityNode(name="Alice", entity_type=EntityType.PERSON)
        bob = EntityNode(name="Bob", entity_type=EntityType.PERSON)
        unified_graph.add_entity(alice)
        unified_graph.add_entity(bob)
        unified_graph.add_entity_relation(alice.id, bob.id, "knows")

        mem1 = self._make_memory("Alice went hiking", mock_embedding_fn, entities=[alice.id])
        mem2 = self._make_memory("Bob went swimming", mock_embedding_fn, entities=[bob.id])
        unified_graph.add_memory(mem1)
        unified_graph.add_memory(mem2)

        query_emb = mock_embedding_fn("what does Alice's friend do")
        results = unified_graph.multi_hop_query(
            start_entities=[alice.id],
            query_embedding=query_emb,
            max_hops=2,
        )
        # Should find at least Alice's direct memory and possibly Bob's through the knows relation
        assert len(results) >= 1

    def test_get_memory(self, unified_graph, mock_embedding_fn):
        mem = self._make_memory("test", mock_embedding_fn)
        unified_graph.add_memory(mem)
        assert unified_graph.get_memory(mem.id) is not None
        assert unified_graph.get_memory("bad-id") is None
