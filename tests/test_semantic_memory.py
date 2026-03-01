"""Tests for SemanticMemoryStore: fact storage with contradiction tracking."""

from zerogmem.memory.semantic import SemanticMemoryStore, Fact


class TestFact:
    """Tests for the Fact dataclass."""

    def test_confirm_increases_confidence(self):
        fact = Fact(subject="Alice", predicate="likes", object="hiking", confidence=0.7)
        fact.confirm("source-2")
        assert abs(fact.confidence - 0.8) < 1e-9
        assert fact.confirmation_count == 2
        assert "source-2" in fact.sources

    def test_confirm_capped_at_one(self):
        fact = Fact(subject="A", predicate="p", object="o", confidence=0.95)
        fact.confirm("s")
        assert fact.confidence == 1.0

    def test_contradict_decreases_confidence(self):
        fact = Fact(subject="Alice", predicate="likes", object="hiking", confidence=0.8)
        fact.contradict("contra-1")
        assert abs(fact.confidence - 0.6) < 1e-9
        assert "contra-1" in fact.contradictions

    def test_contradict_floors_at_01(self):
        fact = Fact(subject="A", predicate="p", object="o", confidence=0.1)
        fact.contradict("s")
        assert fact.confidence == 0.1  # max(0.1, 0.1-0.2) = 0.1

    def test_negate_zeros_confidence(self):
        fact = Fact(subject="Alice", predicate="likes", object="hiking", confidence=0.9)
        fact.negate("neg-source")
        assert fact.negated is True
        assert fact.confidence == 0.0
        assert fact.negation_source == "neg-source"

    def test_is_reliable_positive(self):
        fact = Fact(
            subject="A", predicate="p", object="o",
            confidence=0.8, sources=["s1", "s2"],
        )
        assert fact.is_reliable is True

    def test_is_reliable_false_low_confidence(self):
        fact = Fact(subject="A", predicate="p", object="o", confidence=0.3)
        assert fact.is_reliable is False

    def test_is_reliable_false_negated(self):
        fact = Fact(subject="A", predicate="p", object="o", confidence=0.8, negated=True)
        assert fact.is_reliable is False

    def test_is_reliable_false_more_contradictions(self):
        fact = Fact(
            subject="A", predicate="p", object="o",
            confidence=0.8,
            sources=["s1"],
            contradictions=["c1", "c2"],
        )
        assert fact.is_reliable is False


class TestSemanticMemoryStore:
    """Tests for the SemanticMemoryStore."""

    def test_add_new_fact(self, semantic_store, make_fact):
        fact = make_fact("Alice", "likes", "hiking", category="preference")
        fid, is_new = semantic_store.add_fact(fact)
        assert is_new is True
        assert fid == fact.id
        assert len(semantic_store.facts) == 1

    def test_add_duplicate_merges(self, semantic_store, make_fact):
        f1 = make_fact("Alice", "likes", "hiking")
        f2 = make_fact("Alice", "likes", "hiking")
        fid1, is_new1 = semantic_store.add_fact(f1)
        fid2, is_new2 = semantic_store.add_fact(f2)
        assert is_new1 is True
        assert is_new2 is False
        assert fid2 == fid1  # Merged into existing
        assert len(semantic_store.facts) == 1
        assert f1.confirmation_count == 2

    def test_contradiction_via_contradict_method(self, semantic_store, make_fact):
        # Directly test that the store records contradictions when a fact
        # is explicitly contradicted (the store's _find_similar_fact treats
        # same-SPO facts as duplicates regardless of negation flag, so
        # contradiction detection relies on the Fact.contradict method)
        fact = make_fact("Alice", "likes", "spiders")
        semantic_store.add_fact(fact)
        fact.contradict("contradicting-source")

        contradicted = semantic_store.get_contradicted_facts()
        assert len(contradicted) == 1
        assert contradicted[0].id == fact.id

    def test_get_facts_about(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        semantic_store.add_fact(make_fact("Alice", "lives_in", "NYC"))
        semantic_store.add_fact(make_fact("Bob", "likes", "cooking"))

        results = semantic_store.get_facts_about("Alice")
        assert len(results) == 2

    def test_get_facts_about_with_predicate(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        semantic_store.add_fact(make_fact("Alice", "lives_in", "NYC"))

        results = semantic_store.get_facts_about("Alice", predicate="likes")
        assert len(results) == 1
        assert results[0].object == "hiking"

    def test_get_facts_about_excludes_negated(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        semantic_store.add_fact(make_fact("Alice", "likes", "spiders", negated=True))

        results = semantic_store.get_facts_about("Alice")
        assert len(results) == 1
        assert results[0].object == "hiking"

    def test_get_facts_about_includes_negated(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        semantic_store.add_fact(make_fact("Alice", "likes", "spiders", negated=True))

        results = semantic_store.get_facts_about("Alice", include_negated=True)
        assert len(results) == 2

    def test_get_facts_about_min_confidence(self, semantic_store, make_fact):
        high = make_fact("Alice", "likes", "hiking", confidence=0.9)
        low = make_fact("Alice", "likes", "cooking", confidence=0.2)
        semantic_store.add_fact(high)
        semantic_store.add_fact(low)

        results = semantic_store.get_facts_about("Alice", min_confidence=0.5)
        assert len(results) == 1

    def test_check_negation_found(self, semantic_store, make_fact):
        neg_fact = make_fact("Alice", "likes", "spiders", negated=True)
        semantic_store.add_fact(neg_fact)

        is_negated, fact = semantic_store.check_negation("Alice", "likes", "spiders")
        assert is_negated is True
        assert fact is not None

    def test_check_negation_not_found(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        is_negated, fact = semantic_store.check_negation("Alice", "likes", "hiking")
        assert is_negated is False
        assert fact is None

    def test_add_negation(self, semantic_store):
        fid = semantic_store.add_negation("Alice", "likes", "spiders", "ep-1")
        assert fid is not None
        fact = semantic_store.get_fact(fid)
        assert fact.negated is True
        assert "does NOT" in fact.content

    def test_get_reliable_facts(self, semantic_store, make_fact):
        reliable = make_fact("Alice", "likes", "hiking", confidence=0.8)
        unreliable = make_fact("Alice", "likes", "cooking", confidence=0.3)
        semantic_store.add_fact(reliable)
        semantic_store.add_fact(unreliable)

        results = semantic_store.get_reliable_facts()
        assert len(results) == 1

    def test_get_contradicted_facts(self, semantic_store, make_fact):
        f = make_fact("Alice", "likes", "spiders")
        semantic_store.add_fact(f)
        f.contradict("contra-source")
        results = semantic_store.get_contradicted_facts()
        assert len(results) == 1

    def test_get_negated_facts(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        semantic_store.add_fact(make_fact("Alice", "likes", "spiders", negated=True))
        results = semantic_store.get_negated_facts()
        assert len(results) == 1

    def test_search_similar(self, semantic_store, make_fact, mock_embedding_fn):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        query_emb = mock_embedding_fn("Alice likes hiking")
        results = semantic_store.search_similar(query_emb, top_k=5)
        assert len(results) >= 1

    def test_search_similar_empty(self, semantic_store, mock_embedding_fn):
        query_emb = mock_embedding_fn("anything")
        results = semantic_store.search_similar(query_emb)
        assert results == []

    def test_get_user_profile(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking", category="preference"))
        semantic_store.add_fact(make_fact("Alice", "age", "30", category="attribute"))
        semantic_store.add_fact(make_fact("Alice", "likes", "spiders", negated=True))

        profile = semantic_store.get_user_profile("Alice")
        assert profile["user_id"] == "Alice"
        assert len(profile["preferences"]) == 1
        assert len(profile["attributes"]) == 1
        assert len(profile["dislikes"]) == 1

    def test_get_stats(self, semantic_store, make_fact):
        stats = semantic_store.get_stats()
        assert stats["total_facts"] == 0

        semantic_store.add_fact(make_fact("Alice", "likes", "hiking", category="preference"))
        stats = semantic_store.get_stats()
        assert stats["total_facts"] == 1
        assert stats["unique_subjects"] == 1

    def test_get_facts_by_predicate(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking"))
        semantic_store.add_fact(make_fact("Bob", "likes", "cooking"))
        semantic_store.add_fact(make_fact("Alice", "lives_in", "NYC"))

        results = semantic_store.get_facts_by_predicate("likes")
        assert len(results) == 2

    def test_get_facts_by_category(self, semantic_store, make_fact):
        semantic_store.add_fact(make_fact("Alice", "likes", "hiking", category="preference"))
        semantic_store.add_fact(make_fact("Alice", "age", "30", category="attribute"))

        results = semantic_store.get_facts_by_category("preference")
        assert len(results) == 1
