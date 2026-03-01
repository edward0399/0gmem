"""Tests for SemanticGraph: embedding-based semantic memory."""

import numpy as np

from zerogmem.graph.semantic import SemanticGraph, SemanticNode, SemanticEdge


class TestSemanticGraph:
    """Tests for the SemanticGraph."""

    def _make_node(self, content, embedding_fn, concepts=None):
        return SemanticNode(
            content=content,
            embedding=embedding_fn(content),
            concepts=concepts or [],
        )

    def test_add_node(self, semantic_graph, mock_embedding_fn):
        node = self._make_node("hiking in mountains", mock_embedding_fn, concepts=["hiking"])
        nid = semantic_graph.add_node(node)
        assert nid == node.id
        assert node.id in semantic_graph.nodes

    def test_find_similar(self, semantic_graph, mock_embedding_fn):
        n1 = self._make_node("hiking in mountains", mock_embedding_fn)
        n2 = self._make_node("cooking dinner at home", mock_embedding_fn)
        semantic_graph.add_node(n1)
        semantic_graph.add_node(n2)

        query_emb = mock_embedding_fn("hiking in mountains")
        results = semantic_graph.find_similar(query_emb, top_k=5)
        assert len(results) >= 1
        # The exact same text should have highest similarity
        assert results[0][0].content == "hiking in mountains"
        assert results[0][1] > 0.9  # Near-perfect match

    def test_find_similar_with_threshold(self, semantic_graph, mock_embedding_fn):
        semantic_graph.add_node(self._make_node("hiking", mock_embedding_fn))
        semantic_graph.add_node(self._make_node("cooking", mock_embedding_fn))

        query_emb = mock_embedding_fn("hiking")
        results = semantic_graph.find_similar(query_emb, threshold=0.99)
        # Only the exact match should pass a very high threshold
        assert len(results) >= 1
        assert results[0][0].content == "hiking"

    def test_find_similar_empty(self, semantic_graph, mock_embedding_fn):
        query_emb = mock_embedding_fn("anything")
        results = semantic_graph.find_similar(query_emb)
        assert results == []

    def test_find_by_concept(self, semantic_graph, mock_embedding_fn):
        semantic_graph.add_node(
            self._make_node("alpine hiking", mock_embedding_fn, concepts=["hiking", "alps"])
        )
        semantic_graph.add_node(
            self._make_node("spaghetti recipe", mock_embedding_fn, concepts=["cooking"])
        )

        results = semantic_graph.find_by_concept("hiking")
        assert len(results) == 1
        assert results[0].content == "alpine hiking"

    def test_find_by_concept_missing(self, semantic_graph):
        results = semantic_graph.find_by_concept("nonexistent")
        assert results == []

    def test_find_related_bfs(self, semantic_graph, mock_embedding_fn):
        n1 = self._make_node("A", mock_embedding_fn)
        n2 = self._make_node("B", mock_embedding_fn)
        n3 = self._make_node("C", mock_embedding_fn)
        semantic_graph.add_node(n1)
        semantic_graph.add_node(n2)
        semantic_graph.add_node(n3)

        semantic_graph.add_edge(SemanticEdge(source_id=n1.id, target_id=n2.id, relation="related_to"))
        semantic_graph.add_edge(SemanticEdge(source_id=n2.id, target_id=n3.id, relation="is_a"))

        results = semantic_graph.find_related(n1.id, max_depth=2)
        # Should find n2 at depth 1 and n3 at depth 2
        assert len(results) >= 2

    def test_find_related_with_filter(self, semantic_graph, mock_embedding_fn):
        n1 = self._make_node("A", mock_embedding_fn)
        n2 = self._make_node("B", mock_embedding_fn)
        n3 = self._make_node("C", mock_embedding_fn)
        semantic_graph.add_node(n1)
        semantic_graph.add_node(n2)
        semantic_graph.add_node(n3)

        semantic_graph.add_edge(SemanticEdge(source_id=n1.id, target_id=n2.id, relation="is_a"))
        semantic_graph.add_edge(SemanticEdge(source_id=n1.id, target_id=n3.id, relation="similar_to"))

        results = semantic_graph.find_related(n1.id, relation_filter=["is_a"], max_depth=1)
        assert len(results) == 1

    def test_auto_link_similar(self, semantic_graph, mock_embedding_fn):
        # Add two nodes with identical embeddings (same text)
        n1 = self._make_node("hiking in mountains", mock_embedding_fn)
        n2 = self._make_node("hiking in mountains", mock_embedding_fn)  # Same embedding
        semantic_graph.add_node(n1)
        semantic_graph.add_node(n2)

        edges_created = semantic_graph.auto_link_similar(threshold=0.9)
        assert edges_created >= 1

    def test_update_importance(self, semantic_graph, mock_embedding_fn):
        node = self._make_node("test", mock_embedding_fn)
        semantic_graph.add_node(node)
        assert node.importance == 0.5
        assert node.access_count == 0

        semantic_graph.update_importance(node.id, delta=0.2)
        assert node.importance == 0.7
        assert node.access_count == 1
        assert node.last_accessed is not None

    def test_compute_similarity_zero_norm(self, semantic_graph):
        zero = np.zeros(1536, dtype=np.float32)
        nonzero = np.random.randn(1536).astype(np.float32)
        assert semantic_graph.compute_similarity(zero, nonzero) == 0.0

    def test_get_embedding_matrix(self, semantic_graph, mock_embedding_fn):
        semantic_graph.add_node(self._make_node("a", mock_embedding_fn))
        semantic_graph.add_node(self._make_node("b", mock_embedding_fn))
        mat = semantic_graph.get_embedding_matrix()
        assert mat.shape == (2, 1536)

    def test_get_embedding_matrix_empty(self, semantic_graph):
        mat = semantic_graph.get_embedding_matrix()
        assert len(mat) == 0
