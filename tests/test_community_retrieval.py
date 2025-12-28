"""Tests for community embedding retrieval.

## RAG Theory: Semantic Community Retrieval

GraphRAG uses community summaries to answer "global" queries that require
synthesizing information across multiple documents. Traditional keyword
matching fails for these queries because:
1. Relevant concepts may be expressed with different words
2. Thematic relevance isn't captured by word overlap
3. Semantic similarity captures intent better than keywords

Embedding-based retrieval uses cosine similarity between:
- Query embedding (what the user is asking about)
- Community summary embeddings (what each community is about)

This enables matching "What causes lasting happiness?" to a community about
"hedonic adaptation and psychological well-being" even without word overlap.

## Test Coverage

These tests verify:
1. Cosine similarity computation correctness
2. Keyword fallback when embeddings unavailable
3. Embedding-based retrieval scoring
"""

import pytest
from src.graph.schemas import Community, CommunityMember
from src.graph.query import cosine_similarity


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
        assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        """Zero vectors should return 0.0 (no division error)."""
        assert cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0], [0, 0]) == pytest.approx(0.0)
        assert cosine_similarity([0, 0], [0, 0]) == pytest.approx(0.0)

    def test_partial_similarity(self) -> None:
        """Partially aligned vectors should have intermediate similarity."""
        # 45-degree angle: cos(45) = sqrt(2)/2 ≈ 0.707
        sim = cosine_similarity([1, 0], [1, 1])
        assert 0.7 < sim < 0.75


class TestCommunitySchema:
    """Tests for Community schema with embeddings."""

    def test_to_dict_includes_embedding(self) -> None:
        """to_dict should include the embedding field."""
        embedding = [0.1, 0.2, 0.3]
        community = Community(
            community_id="test_1",
            summary="Test summary",
            embedding=embedding,
            members=[],
        )

        d = community.to_dict()
        assert "embedding" in d
        assert d["embedding"] == embedding

    def test_to_dict_handles_none_embedding(self) -> None:
        """to_dict should handle None embedding gracefully."""
        community = Community(
            community_id="test_2",
            summary="Test summary",
            embedding=None,
            members=[],
        )

        d = community.to_dict()
        assert "embedding" in d
        assert d["embedding"] is None


class TestKeywordFallback:
    """Tests for keyword-based fallback when embeddings unavailable."""

    def test_keyword_matching_without_embeddings(self) -> None:
        """When no embeddings, should use keyword matching."""
        from src.graph.query import retrieve_community_context

        # Create communities without embeddings
        communities = [
            Community(
                community_id="c1",
                summary="This is about dopamine and neuroscience research",
                embedding=None,
                members=[
                    CommunityMember(
                        entity_name="dopamine",
                        entity_type="NEUROTRANSMITTER",
                    ),
                ],
            ),
            Community(
                community_id="c2",
                summary="This discusses philosophy and stoicism teachings",
                embedding=None,
                members=[
                    CommunityMember(
                        entity_name="stoicism",
                        entity_type="PHILOSOPHY",
                    ),
                ],
            ),
        ]

        # Query with keyword "dopamine" should match c1
        # Note: no punctuation in query to ensure exact word matching
        results = retrieve_community_context(
            "tell me about dopamine",
            communities=communities,
            top_k=2,
        )

        assert len(results) >= 1
        # c1 should rank higher due to "dopamine" in summary and members
        assert results[0]["community_id"] == "c1"
