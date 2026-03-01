"""Tests for EpisodicMemory: episode storage and retrieval."""

from datetime import datetime, timedelta

from zerogmem.memory.episodic import Episode, EpisodeMessage


class TestEpisode:
    """Tests for the Episode dataclass."""

    def test_duration_with_end_time(self):
        ep = Episode(
            start_time=datetime(2024, 6, 1, 10, 0),
            end_time=datetime(2024, 6, 1, 11, 30),
        )
        assert ep.duration == timedelta(hours=1, minutes=30)

    def test_duration_none_without_end_time(self):
        ep = Episode(start_time=datetime(2024, 6, 1, 10, 0))
        assert ep.duration is None

    def test_message_count(self):
        ep = Episode()
        assert ep.message_count == 0
        ep.messages = [EpisodeMessage(speaker="A", content="hi")]
        assert ep.message_count == 1

    def test_get_full_text(self):
        ep = Episode()
        ep.messages = [
            EpisodeMessage(speaker="Alice", content="Hello"),
            EpisodeMessage(speaker="Bob", content="Hi there"),
        ]
        text = ep.get_full_text()
        assert "Alice: Hello" in text
        assert "Bob: Hi there" in text

    def test_add_message_updates_end_time_and_participants(self):
        ep = Episode()
        msg = EpisodeMessage(
            speaker="Alice",
            content="Hello",
            timestamp=datetime(2024, 6, 1, 12, 0),
        )
        ep.add_message(msg)
        assert ep.end_time == datetime(2024, 6, 1, 12, 0)
        assert "Alice" in ep.participant_names

    def test_add_message_no_duplicate_participants(self):
        ep = Episode()
        ep.add_message(EpisodeMessage(speaker="Alice", content="hi"))
        ep.add_message(EpisodeMessage(speaker="Alice", content="bye"))
        assert ep.participant_names.count("Alice") == 1

    def test_mark_retrieved(self):
        ep = Episode()
        assert ep.retrieval_count == 0
        ep.mark_retrieved()
        assert ep.retrieval_count == 1
        assert ep.last_retrieved is not None

    def test_get_text_window(self):
        ep = Episode()
        for i in range(5):
            ep.messages.append(EpisodeMessage(speaker="A", content=f"msg{i}"))
        window = ep.get_text_window(1, 3)
        assert "msg1" in window
        assert "msg2" in window
        assert "msg0" not in window


class TestEpisodicMemory:
    """Tests for the EpisodicMemory store."""

    def test_add_episode(self, episodic_memory, make_episode):
        ep = make_episode(participants=["Alice"])
        eid = episodic_memory.add_episode(ep)
        assert eid == ep.id
        assert len(episodic_memory.episodes) == 1

    def test_get_episode(self, episodic_memory, make_episode):
        ep = make_episode(participants=["Alice"])
        episodic_memory.add_episode(ep)
        retrieved = episodic_memory.get_episode(ep.id)
        assert retrieved is not None
        assert retrieved.id == ep.id
        assert retrieved.retrieval_count == 1

    def test_get_episode_no_mark(self, episodic_memory, make_episode):
        ep = make_episode()
        episodic_memory.add_episode(ep)
        retrieved = episodic_memory.get_episode(ep.id, mark_retrieved=False)
        assert retrieved is not None
        assert retrieved.retrieval_count == 0

    def test_get_nonexistent_episode(self, episodic_memory):
        assert episodic_memory.get_episode("bad-id") is None

    def test_get_by_time_range(self, episodic_memory, make_episode):
        ep1 = make_episode(start_time=datetime(2024, 6, 1, 10, 0))
        ep2 = make_episode(start_time=datetime(2024, 6, 5, 10, 0))
        # Put the third episode far enough away that date iteration won't reach it
        ep3 = make_episode(start_time=datetime(2024, 8, 1, 10, 0))
        for ep in [ep1, ep2, ep3]:
            episodic_memory.add_episode(ep)

        results = episodic_memory.get_by_time_range(
            start=datetime(2024, 6, 1),
            end=datetime(2024, 6, 30),
        )
        assert len(results) == 2

    def test_get_by_time_range_with_participants(self, episodic_memory, make_episode):
        ep1 = make_episode(
            participants=["Alice"],
            start_time=datetime(2024, 6, 1, 10, 0),
        )
        ep2 = make_episode(
            participants=["Bob"],
            start_time=datetime(2024, 6, 2, 10, 0),
        )
        episodic_memory.add_episode(ep1)
        episodic_memory.add_episode(ep2)

        results = episodic_memory.get_by_time_range(
            start=datetime(2024, 6, 1),
            end=datetime(2024, 6, 30),
            participants=["Alice"],
        )
        assert len(results) == 1

    def test_get_by_participant(self, episodic_memory, make_episode):
        ep = make_episode(participants=["Alice"])
        episodic_memory.add_episode(ep)
        results = episodic_memory.get_by_participant("Alice")
        assert len(results) == 1

    def test_get_by_topic(self, episodic_memory, make_episode):
        ep = make_episode(topics=["hiking"])
        episodic_memory.add_episode(ep)
        results = episodic_memory.get_by_topic("hiking")
        assert len(results) == 1
        assert episodic_memory.get_by_topic("cooking") == []

    def test_get_by_session(self, episodic_memory, make_episode):
        sid = "session-42"
        ep = make_episode(session_id=sid)
        episodic_memory.add_episode(ep)
        results = episodic_memory.get_by_session(sid)
        assert len(results) == 1

    def test_search_similar(self, episodic_memory, make_episode, mock_embedding_fn):
        ep = make_episode(participants=["Alice"])
        episodic_memory.add_episode(ep)
        query_emb = mock_embedding_fn("episode summary")
        results = episodic_memory.search_similar(query_emb, top_k=5)
        assert len(results) >= 1
        assert results[0][1] > 0  # score > 0

    def test_search_similar_empty(self, episodic_memory, mock_embedding_fn):
        query_emb = mock_embedding_fn("anything")
        results = episodic_memory.search_similar(query_emb)
        assert results == []

    def test_get_recent(self, episodic_memory, make_episode):
        for i in range(3):
            ep = make_episode(
                start_time=datetime(2024, 6, 1 + i, 10, 0),
            )
            episodic_memory.add_episode(ep)
        results = episodic_memory.get_recent(limit=2)
        assert len(results) == 2
        # Most recent first
        assert results[0].start_time > results[1].start_time

    def test_get_most_accessed(self, episodic_memory, make_episode):
        ep1 = make_episode()
        ep2 = make_episode()
        episodic_memory.add_episode(ep1)
        episodic_memory.add_episode(ep2)
        # Access ep2 more times
        for _ in range(3):
            ep2.mark_retrieved()
        results = episodic_memory.get_most_accessed(limit=2)
        assert results[0].retrieval_count >= results[1].retrieval_count

    def test_archive_episode(self, episodic_memory, make_episode):
        ep = make_episode()
        episodic_memory.add_episode(ep)
        result = episodic_memory.archive_episode(ep.id, "s3://archive/ep1")
        assert result is True
        assert ep.archived is True
        assert ep.archive_ref == "s3://archive/ep1"

    def test_archive_nonexistent(self, episodic_memory):
        assert episodic_memory.archive_episode("bad-id", "ref") is False

    def test_get_stats(self, episodic_memory, make_episode):
        stats = episodic_memory.get_stats()
        assert stats["total_episodes"] == 0

        ep = make_episode(
            participants=["Alice"],
            topics=["hiking"],
            messages=[("Alice", "I love hiking")],
        )
        episodic_memory.add_episode(ep)
        stats = episodic_memory.get_stats()
        assert stats["total_episodes"] == 1
        assert stats["total_messages"] == 1
        assert stats["unique_participants"] >= 1
