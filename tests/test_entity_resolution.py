"""Tests for entity name normalization.

## RAG Theory: Entity Resolution

Entity resolution (also called entity matching or deduplication) ensures that
different textual mentions of the same entity map to a single node in the
knowledge graph. For example, "The Dopamine", "dopamine", and "DOPAMINE"
should all resolve to the same entity.

Good normalization is critical for GraphRAG because:
1. Prevents duplicate nodes from fragmenting entity relationships
2. Enables proper community detection by Leiden algorithm
3. Improves retrieval by consolidating entity references

## Test Coverage

These tests verify the enhanced normalization handles:
- Case normalization (UPPERCASE -> lowercase)
- Unicode normalization (cafe with accents -> cafe)
- Leading/trailing stopword removal (The X -> X)
- Punctuation stripping (X's -> Xs)
- Whitespace normalization (spaced  out -> spaced out)
"""

import pytest
from src.graph.schemas import GraphEntity


@pytest.mark.parametrize("input_name,expected", [
    # Basic case normalization
    ("dopamine", "dopamine"),
    ("DOPAMINE", "dopamine"),
    ("Dopamine", "dopamine"),

    # Leading stopword removal
    ("The Dopamine", "dopamine"),
    ("A prefrontal cortex", "prefrontal cortex"),
    ("An amygdala", "amygdala"),
    ("The Art of War", "art of war"),  # Middle stopwords preserved

    # Trailing stopword removal
    ("Prefrontal Cortex, The", "prefrontal cortex"),
    ("Mind and", "mind"),

    # Unicode normalization (NFKC)
    ("cafe", "cafe"),

    # Punctuation stripping
    ("Marcus Aurelius' Meditations", "marcus aurelius meditations"),
    ("it's", "its"),
    ("self-control", "selfcontrol"),

    # Whitespace normalization
    ("  spaced  out  ", "spaced out"),
    ("multiple   spaces", "multiple spaces"),

    # Edge cases
    ("the", ""),  # Only stopword -> empty
    ("a", ""),
    # Empty string not tested - Pydantic requires min_length=1
    ("   ", ""),  # Only whitespace

    # Combined transformations
    ("The PREFRONTAL Cortex", "prefrontal cortex"),
    ("  An   Example  ", "example"),
])
def test_normalized_name(input_name: str, expected: str) -> None:
    """Test entity name normalization for various inputs."""
    entity = GraphEntity(name=input_name, entity_type="TEST")
    assert entity.normalized_name() == expected


def test_normalization_consistency() -> None:
    """Verify that different forms of the same entity normalize identically."""
    variants = [
        "The Dopamine",
        "dopamine",
        "DOPAMINE",
        "  dopamine  ",
        "the dopamine",
    ]

    normalized = [
        GraphEntity(name=v, entity_type="TEST").normalized_name()
        for v in variants
    ]

    # All should normalize to the same value
    assert len(set(normalized)) == 1
    assert normalized[0] == "dopamine"


def test_normalization_distinguishes_different_entities() -> None:
    """Verify that different entities remain distinct after normalization."""
    entities = [
        "dopamine",
        "serotonin",
        "cortisol",
        "prefrontal cortex",
        "amygdala",
    ]

    normalized = [
        GraphEntity(name=e, entity_type="TEST").normalized_name()
        for e in entities
    ]

    # All should remain distinct
    assert len(set(normalized)) == len(entities)
