"""Tests for TemporalGraph: temporal reasoning with Allen's Interval Algebra."""

from datetime import datetime, timedelta

from zerogmem.graph.temporal import (
    TemporalNode,
    TemporalRelation,
    TimeInterval,
)


class TestTimeInterval:
    """Tests for the TimeInterval dataclass."""

    def test_is_point_no_end(self):
        ti = TimeInterval(start=datetime(2024, 6, 1))
        assert ti.is_point is True

    def test_is_point_same_start_end(self):
        t = datetime(2024, 6, 1, 12, 0)
        ti = TimeInterval(start=t, end=t)
        assert ti.is_point is True

    def test_is_not_point(self):
        ti = TimeInterval(
            start=datetime(2024, 6, 1, 10, 0),
            end=datetime(2024, 6, 1, 12, 0),
        )
        assert ti.is_point is False

    def test_duration(self):
        ti = TimeInterval(
            start=datetime(2024, 6, 1, 10, 0),
            end=datetime(2024, 6, 1, 12, 0),
        )
        assert ti.duration == timedelta(hours=2)

    def test_duration_none_for_point(self):
        ti = TimeInterval(start=datetime(2024, 6, 1))
        assert ti.duration is None

    def test_contains_time_within(self):
        ti = TimeInterval(
            start=datetime(2024, 6, 1, 10, 0),
            end=datetime(2024, 6, 1, 14, 0),
        )
        assert ti.contains_time(datetime(2024, 6, 1, 12, 0)) is True

    def test_contains_time_outside(self):
        ti = TimeInterval(
            start=datetime(2024, 6, 1, 10, 0),
            end=datetime(2024, 6, 1, 14, 0),
        )
        assert ti.contains_time(datetime(2024, 6, 1, 16, 0)) is False

    def test_contains_time_open_ended(self):
        ti = TimeInterval(start=datetime(2024, 6, 1, 10, 0))
        assert ti.contains_time(datetime(2024, 12, 1)) is True
        assert ti.contains_time(datetime(2024, 1, 1)) is False

    def test_overlaps_with_overlapping(self):
        a = TimeInterval(datetime(2024, 6, 1, 10, 0), datetime(2024, 6, 1, 14, 0))
        b = TimeInterval(datetime(2024, 6, 1, 12, 0), datetime(2024, 6, 1, 16, 0))
        assert a.overlaps_with(b) is True

    def test_overlaps_with_disjoint(self):
        a = TimeInterval(datetime(2024, 6, 1, 10, 0), datetime(2024, 6, 1, 12, 0))
        b = TimeInterval(datetime(2024, 6, 1, 14, 0), datetime(2024, 6, 1, 16, 0))
        assert a.overlaps_with(b) is False

    def test_overlaps_open_ended(self):
        a = TimeInterval(start=datetime(2024, 6, 1))
        b = TimeInterval(start=datetime(2024, 6, 2))
        assert a.overlaps_with(b) is True

    def test_contains_time_at_boundary(self):
        ti = TimeInterval(
            start=datetime(2024, 6, 1, 10, 0),
            end=datetime(2024, 6, 1, 14, 0),
        )
        assert ti.contains_time(datetime(2024, 6, 1, 10, 0)) is True  # start
        assert ti.contains_time(datetime(2024, 6, 1, 14, 0)) is True  # end


class TestTemporalGraph:
    """Tests for the TemporalGraph."""

    def _make_node(self, content, start, end=None, entities=None):
        return TemporalNode(
            content=content,
            event_time=TimeInterval(start=start, end=end),
            entities=entities or [],
        )

    def test_add_node(self, temporal_graph):
        node = self._make_node("event A", datetime(2024, 6, 1, 10, 0))
        nid = temporal_graph.add_node(node)
        assert nid == node.id
        assert node.id in temporal_graph.nodes

    def test_get_node(self, temporal_graph):
        node = self._make_node("event A", datetime(2024, 6, 1, 10, 0))
        temporal_graph.add_node(node)
        retrieved = temporal_graph.get_node(node.id)
        assert retrieved is not None
        assert retrieved.content == "event A"

    def test_compute_relation_before(self, temporal_graph):
        a = self._make_node("A", datetime(2024, 6, 1, 10, 0), datetime(2024, 6, 1, 11, 0))
        b = self._make_node("B", datetime(2024, 6, 1, 14, 0), datetime(2024, 6, 1, 15, 0))
        rel = temporal_graph.compute_relation(a, b)
        assert rel == TemporalRelation.BEFORE

    def test_compute_relation_after(self, temporal_graph):
        a = self._make_node("A", datetime(2024, 6, 1, 14, 0), datetime(2024, 6, 1, 15, 0))
        b = self._make_node("B", datetime(2024, 6, 1, 10, 0), datetime(2024, 6, 1, 11, 0))
        rel = temporal_graph.compute_relation(a, b)
        assert rel == TemporalRelation.AFTER

    def test_compute_relation_equals(self, temporal_graph):
        t1 = datetime(2024, 6, 1, 10, 0)
        t2 = datetime(2024, 6, 1, 12, 0)
        a = self._make_node("A", t1, t2)
        b = self._make_node("B", t1, t2)
        rel = temporal_graph.compute_relation(a, b)
        assert rel == TemporalRelation.EQUALS

    def test_compute_relation_during(self, temporal_graph):
        # A is fully inside B
        a = self._make_node("A", datetime(2024, 6, 1, 11, 0), datetime(2024, 6, 1, 13, 0))
        b = self._make_node("B", datetime(2024, 6, 1, 10, 0), datetime(2024, 6, 1, 14, 0))
        rel = temporal_graph.compute_relation(a, b)
        assert rel == TemporalRelation.DURING

    def test_events_at(self, temporal_graph):
        node = self._make_node(
            "lunch",
            datetime(2024, 6, 1, 12, 0),
            datetime(2024, 6, 1, 13, 0),
        )
        temporal_graph.add_node(node)
        results = temporal_graph.events_at(datetime(2024, 6, 1, 12, 30))
        assert len(results) >= 1

    def test_events_in_range(self, temporal_graph):
        n1 = self._make_node("morning", datetime(2024, 6, 1, 9, 0), datetime(2024, 6, 1, 10, 0))
        n2 = self._make_node("afternoon", datetime(2024, 6, 1, 14, 0), datetime(2024, 6, 1, 15, 0))
        n3 = self._make_node("next day", datetime(2024, 6, 2, 9, 0), datetime(2024, 6, 2, 10, 0))
        for n in [n1, n2, n3]:
            temporal_graph.add_node(n)

        results = temporal_graph.events_in_range(
            datetime(2024, 6, 1, 0, 0),
            datetime(2024, 6, 1, 23, 59),
        )
        assert len(results) == 2

    def test_events_before(self, temporal_graph):
        early = self._make_node("early", datetime(2024, 6, 1, 8, 0), datetime(2024, 6, 1, 9, 0))
        late = self._make_node("late", datetime(2024, 6, 1, 14, 0), datetime(2024, 6, 1, 15, 0))
        temporal_graph.add_node(early)
        temporal_graph.add_node(late)
        results = temporal_graph.events_before(late)
        assert any(n.content == "early" for n in results)

    def test_events_after(self, temporal_graph):
        early = self._make_node("early", datetime(2024, 6, 1, 8, 0), datetime(2024, 6, 1, 9, 0))
        late = self._make_node("late", datetime(2024, 6, 1, 14, 0), datetime(2024, 6, 1, 15, 0))
        temporal_graph.add_node(early)
        temporal_graph.add_node(late)
        results = temporal_graph.events_after(early)
        assert any(n.content == "late" for n in results)

    def test_temporal_chain(self, temporal_graph):
        entity_id = "alice"
        for i, hour in enumerate([8, 10, 14]):
            n = self._make_node(
                f"event_{i}",
                datetime(2024, 6, 1, hour, 0),
                datetime(2024, 6, 1, hour + 1, 0),
                entities=[entity_id],
            )
            temporal_graph.add_node(n)
        chain = temporal_graph.temporal_chain(entity_id)
        assert len(chain) == 3
        # Should be chronological
        for i in range(len(chain) - 1):
            assert chain[i].event_time.start <= chain[i + 1].event_time.start

    def test_find_by_entities(self, temporal_graph):
        n1 = self._make_node("trip", datetime(2024, 6, 1), entities=["alice", "bob"])
        n2 = self._make_node("solo", datetime(2024, 6, 2), entities=["charlie"])
        temporal_graph.add_node(n1)
        temporal_graph.add_node(n2)

        results = temporal_graph.find_by_entities(["alice"])
        assert len(results) == 1
        assert results[0].content == "trip"

    def test_get_node_nonexistent(self, temporal_graph):
        assert temporal_graph.get_node("bad-id") is None
