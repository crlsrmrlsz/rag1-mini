"""Pydantic schemas for query preprocessing LLM responses.

## RAG Theory: Query Preprocessing Schemas

These schemas define the expected structure of LLM responses during
query preprocessing. Each schema corresponds to a specific preprocessing
function and ensures type-safe extraction of principles, queries, and
decomposition results.

Benefits of schema-based parsing:
1. **Guaranteed types** - No more isinstance() checks or .get() fallbacks
2. **Clear contracts** - Schema documents exactly what LLM should return
3. **Validation errors** - Descriptive errors instead of silent failures
4. **IDE support** - Autocomplete and type hints for response fields

## Library Usage

Uses Pydantic v2 BaseModel with:
- Literal types for enum-like constraints
- Field() for defaults and descriptions
- model_validate_json() for parsing

## Data Flow

1. System prompt instructs LLM to return specific JSON structure
2. OpenRouter enforces schema via response_format (if supported)
3. Pydantic validates and parses response into typed object
4. Calling code accesses fields with full type safety
"""

from typing import List

from pydantic import BaseModel, Field


class PrincipleExtraction(BaseModel):
    """Extracted concepts from a query - cross-domain aware.

    Used by: extract_principles()

    Contains the semantic decomposition of a query into domain-specific
    concepts for a dual-domain corpus (neuroscience + philosophy).
    This is the first step of multi-query generation, identifying
    the core concepts that should inform query generation.

    Example response:
        {
            "core_theme": "fear regulation and courage",
            "mechanism_terms": ["amygdala", "fear extinction", "stress response", "cortisol"],
            "principle_terms": ["Stoic courage", "acceptance", "premeditatio malorum"],
            "bridge_terms": ["resilience", "emotional regulation", "self-mastery"]
        }
    """

    core_theme: str = Field(description="The fundamental subject (3-5 words)")
    mechanism_terms: List[str] = Field(
        default_factory=list,
        description="Brain/biological vocabulary: brain regions, neurotransmitters, cognitive processes",
    )
    principle_terms: List[str] = Field(
        default_factory=list,
        description="Philosophical vocabulary: virtues, practices, wisdom concepts",
    )
    bridge_terms: List[str] = Field(
        default_factory=list,
        description="Cross-domain vocabulary that appears in both domains",
    )


class GeneratedQuery(BaseModel):
    """A single generated search query with its type.

    Part of multi-query generation. Each query targets a specific
    domain of the knowledge base for diverse cross-domain retrieval.

    Types:
    - mechanism: Brain/biological processes (targets neuroscience texts)
    - principle: Wisdom/guidance (targets philosophical texts)
    - synthesis: Combined mechanism + principle vocabulary (bridges domains)
    - accessible: Everyday language (targets introductory passages)
    """

    type: str = Field(
        description="Query category: mechanism, principle, synthesis, accessible"
    )
    query: str = Field(description="The search query text (10-15 words)")


class MultiQueryResult(BaseModel):
    """Result of multi-query generation.

    Used by: generate_multi_queries()

    Contains 4 targeted queries for RRF (Reciprocal Rank Fusion) merging.
    Each query targets a different domain (neuroscience/philosophy/bridge),
    combined using RRF to produce a diverse, cross-domain result set.

    Example response:
        {
            "queries": [
                {"type": "mechanism", "query": "amygdala fear extinction cortisol stress..."},
                {"type": "principle", "query": "Stoic courage acceptance premeditatio..."},
                {"type": "synthesis", "query": "fear regulation emotional resilience..."},
                {"type": "accessible", "query": "overcoming fear practical techniques..."}
            ]
        }
    """

    queries: List[GeneratedQuery] = Field(
        description="List of 4 generated search queries targeting different domains"
    )


class DecompositionResult(BaseModel):
    """Result of query decomposition for complex queries.

    Used by: decompose_query()

    Contains 2-4 sub-questions that can be answered independently.
    Each sub-question is used for separate retrieval, with results
    merged using RRF. This handles complex comparison or multi-aspect
    questions that span multiple domains.

    Example for "Compare Stoic and Buddhist views on suffering":
        {
            "sub_questions": [
                "What is the Stoic view on suffering and how to overcome it?",
                "What is the Buddhist teaching on suffering and its cessation?",
                "How do Stoic and Buddhist approaches to suffering differ?"
            ],
            "reasoning": "The question asks for comparison, so we need each
                          tradition's view separately, then a synthesis."
        }
    """

    sub_questions: List[str] = Field(
        description="Self-contained sub-questions for independent retrieval"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decomposition approach",
    )
