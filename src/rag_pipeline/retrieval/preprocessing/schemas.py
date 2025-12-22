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
    """Extracted principles and concepts from a query.

    Used by: extract_principles()

    Contains the semantic decomposition of a query into domain-agnostic
    concepts. This is the first step of multi-query generation, identifying
    the core concepts that should inform query generation.

    Example response:
        {
            "core_topic": "procrastination and self-control",
            "primary_concepts": ["temporal discounting", "motivation", "willpower"],
            "secondary_concepts": ["self-regulation", "delay of gratification"],
            "related_terms": ["avoidance", "task aversion", "impulsivity"]
        }
    """

    core_topic: str = Field(description="The fundamental subject of the query")
    primary_concepts: List[str] = Field(
        default_factory=list,
        description="Specific mechanisms, theories, or frameworks",
    )
    secondary_concepts: List[str] = Field(
        default_factory=list,
        description="Related ideas, schools of thought, or approaches",
    )
    related_terms: List[str] = Field(
        default_factory=list,
        description="Vocabulary likely to appear in relevant passages",
    )


class GeneratedQuery(BaseModel):
    """A single generated search query with its type.

    Part of multi-query generation. Each query targets a specific
    aspect of the knowledge base for diverse retrieval.

    Types:
    - technical: Specific mechanisms, processes, or terminology
    - conceptual: Frameworks, theories, or abstract ideas
    - applied: Practical applications or real-world scenarios
    - broad: Core topic in accessible language
    """

    type: str = Field(
        description="Query category: technical, conceptual, applied, broad"
    )
    query: str = Field(description="The search query text (8-15 words)")


class MultiQueryResult(BaseModel):
    """Result of multi-query generation.

    Used by: generate_multi_queries()

    Contains 4 targeted queries for RRF (Reciprocal Rank Fusion) merging.
    Each query retrieves different relevant chunks, which are then
    combined using RRF to produce a diverse, comprehensive result set.

    Example response:
        {
            "queries": [
                {"type": "technical", "query": "temporal discounting reward delay..."},
                {"type": "conceptual", "query": "self-regulation willpower theory..."},
                {"type": "applied", "query": "practical strategies overcoming..."},
                {"type": "broad", "query": "why do we procrastinate..."}
            ]
        }
    """

    queries: List[GeneratedQuery] = Field(
        description="List of 4 generated search queries"
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
