"""Tests for EntityGraph: entity tracking and relationship management."""

from zerogmem.graph.entity import (
    EntityEdge,
    EntityNode,
    EntityType,
)


class TestEntityNode:
    """Tests for the EntityNode dataclass."""

    def test_matches_name_exact(self):
        node = EntityNode(name="Alice")
        assert node.matches_name("Alice") is True
        assert node.matches_name("alice") is True  # case-insensitive

    def test_matches_name_alias(self):
        node = EntityNode(name="Alice Smith", aliases=["Ali", "A.S."])
        assert node.matches_name("Ali") is True
        assert node.matches_name("a.s.") is True

    def test_matches_name_partial(self):
        node = EntityNode(name="Alice Smith")
        assert node.matches_name("Alice") is True  # substring

    def test_matches_name_no_match(self):
        node = EntityNode(name="Alice")
        assert node.matches_name("Bob") is False


class TestEntityGraph:
    """Tests for the EntityGraph."""

    def _make_node(self, name, entity_type=EntityType.PERSON, aliases=None):
        return EntityNode(name=name, entity_type=entity_type, aliases=aliases or [])

    def test_add_node(self, entity_graph):
        node = self._make_node("Alice")
        nid = entity_graph.add_node(node)
        assert nid == node.id
        assert node.id in entity_graph.nodes

    def test_add_edge(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)

        edge = EntityEdge(source_id=alice.id, target_id=bob.id, relation="knows")
        eid = entity_graph.add_edge(edge)
        assert eid == edge.id

    def test_find_by_name_exact(self, entity_graph):
        alice = self._make_node("Alice")
        entity_graph.add_node(alice)
        results = entity_graph.find_by_name("alice")
        assert len(results) == 1
        assert results[0].name == "Alice"

    def test_find_by_name_fuzzy(self, entity_graph):
        alice = self._make_node("Alice Smith")
        entity_graph.add_node(alice)
        results = entity_graph.find_by_name("Alice", fuzzy=True)
        assert len(results) >= 1

    def test_find_by_name_no_fuzzy(self, entity_graph):
        alice = self._make_node("Alice Smith")
        entity_graph.add_node(alice)
        results = entity_graph.find_by_name("Bob", fuzzy=False)
        assert len(results) == 0

    def test_find_by_type(self, entity_graph):
        alice = self._make_node("Alice", EntityType.PERSON)
        nyc = self._make_node("NYC", EntityType.LOCATION)
        entity_graph.add_node(alice)
        entity_graph.add_node(nyc)

        persons = entity_graph.find_by_type(EntityType.PERSON)
        assert len(persons) == 1
        assert persons[0].name == "Alice"

    def test_get_relations(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)

        edge = EntityEdge(source_id=alice.id, target_id=bob.id, relation="knows")
        entity_graph.add_edge(edge)

        relations = entity_graph.get_relations(alice.id)
        assert len(relations) == 1
        node, rel_edge = relations[0]
        assert node.name == "Bob"
        assert rel_edge.relation == "knows"

    def test_get_relations_with_filter(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        nyc = self._make_node("NYC", EntityType.LOCATION)
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)
        entity_graph.add_node(nyc)

        entity_graph.add_edge(EntityEdge(source_id=alice.id, target_id=bob.id, relation="knows"))
        entity_graph.add_edge(EntityEdge(source_id=alice.id, target_id=nyc.id, relation="lives_in"))

        relations = entity_graph.get_relations(alice.id, relation_filter=["knows"])
        assert len(relations) == 1
        assert relations[0][1].relation == "knows"

    def test_get_relations_exclude_negated(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)

        entity_graph.add_edge(
            EntityEdge(source_id=alice.id, target_id=bob.id, relation="likes", negated=True)
        )

        relations = entity_graph.get_relations(alice.id, include_negated=False)
        assert len(relations) == 0

    def test_has_relation_positive(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)
        entity_graph.add_edge(EntityEdge(source_id=alice.id, target_id=bob.id, relation="knows"))

        exists, negated = entity_graph.has_relation(alice.id, bob.id, "knows")
        assert exists is True
        assert negated is False

    def test_has_relation_negated(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)
        entity_graph.add_negative_relation(alice.id, bob.id, "likes")

        exists, negated = entity_graph.has_relation(alice.id, bob.id, "likes")
        assert exists is True
        assert negated is True

    def test_has_relation_not_found(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)

        exists, negated = entity_graph.has_relation(alice.id, bob.id, "knows")
        assert exists is False
        assert negated is None

    def test_add_negative_relation(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)

        eid = entity_graph.add_negative_relation(alice.id, bob.id, "likes", evidence=["ep-1"])
        edge = entity_graph.edges[eid]
        assert edge.negated is True
        assert "ep-1" in edge.evidence

    def test_find_path(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        charlie = self._make_node("Charlie")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)
        entity_graph.add_node(charlie)

        entity_graph.add_edge(EntityEdge(source_id=alice.id, target_id=bob.id, relation="knows"))
        entity_graph.add_edge(EntityEdge(source_id=bob.id, target_id=charlie.id, relation="knows"))

        paths = entity_graph.find_path(alice.id, charlie.id, max_hops=3)
        assert len(paths) >= 1

    def test_find_path_no_path(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)
        # No edge
        paths = entity_graph.find_path(alice.id, bob.id)
        assert paths == []

    def test_merge_entities(self, entity_graph):
        alice1 = self._make_node("Alice", aliases=[])
        alice2 = self._make_node("Ali", aliases=[])
        bob = self._make_node("Bob")
        entity_graph.add_node(alice1)
        entity_graph.add_node(alice2)
        entity_graph.add_node(bob)

        entity_graph.add_edge(EntityEdge(source_id=alice2.id, target_id=bob.id, relation="knows"))

        merged_id = entity_graph.merge_entities([alice1.id, alice2.id], primary_id=alice1.id)
        assert merged_id == alice1.id
        assert alice2.id not in entity_graph.nodes
        assert "Ali" in alice1.aliases

    def test_get_entity_profile(self, entity_graph):
        alice = self._make_node("Alice")
        bob = self._make_node("Bob")
        entity_graph.add_node(alice)
        entity_graph.add_node(bob)
        entity_graph.add_edge(EntityEdge(source_id=alice.id, target_id=bob.id, relation="knows"))

        profile = entity_graph.get_entity_profile(alice.id)
        assert profile["name"] == "Alice"
        assert len(profile["relations"]) == 1

    def test_get_node_nonexistent(self, entity_graph):
        assert entity_graph.get_node("bad-id") is None
