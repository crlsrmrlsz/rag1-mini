"""Query preprocessing for improved RAG retrieval.

## RAG Theory: Query Preprocessing

Query preprocessing improves retrieval by transforming queries before searching.
Each preprocessing strategy applies its transformation directly to any query,
following the original research papers' design.

Techniques implemented:
1. **Step-back prompting** abstracts questions into broader concepts
   - Paper: "Take a Step Back" (Google DeepMind, 2023) showed +27% on multi-hop
2. **Multi-query generation** creates diverse queries for RRF merging
   - Paper: Query Decomposition showed +36.7% MRR@10 improvement
3. **Query decomposition** breaks complex questions into sub-queries

## Library Usage

Uses OpenRouter API via `requests` for LLM calls.
Pydantic schemas ensure structured LLM outputs.

## Data Flow

1. User selects preprocessing strategy (none, step_back, multi_query, decomposition)
2. Strategy transforms query directly (no classification routing)
3. Return PreprocessedQuery with original, transformed, and metadata
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import requests
from pydantic import ValidationError as PydanticValidationError

from src.config import PREPROCESSING_MODEL, CORPUS_TOPICS
from src.shared.files import setup_logging
from src.shared.openrouter_client import call_chat_completion, call_structured_completion
from src.rag_pipeline.retrieval.preprocessing.schemas import (
    PrincipleExtraction,
    MultiQueryResult,
    DecompositionResult,
)

logger = setup_logging(__name__)


@dataclass
class PreprocessedQuery:
    """Result of query preprocessing.

    Attributes:
        original_query: The user's original query text.
        search_query: The query to use for retrieval (may be transformed).
        strategy_used: The preprocessing strategy that was applied.
        preprocessing_time_ms: Time taken for preprocessing in milliseconds.
        model: Model ID used for preprocessing (for logging).
        step_back_query: The transformed broader query (step_back strategy).
        step_back_response: Raw step-back response from LLM.
        sub_queries: Decomposed sub-questions (decomposition strategy).
        generated_queries: List of {type, query} dicts for multi-query retrieval.
        principle_extraction: Extracted principles (multi_query strategy).
        decomposition_response: Raw decomposition response from LLM.
    """

    original_query: str
    search_query: str
    strategy_used: str = ""
    preprocessing_time_ms: float = 0.0
    model: str = ""
    # Step-back strategy fields
    step_back_query: Optional[str] = None
    step_back_response: Optional[str] = None
    # Multi-query strategy fields
    generated_queries: List[Dict[str, str]] = field(default_factory=list)
    principle_extraction: Optional[Dict[str, Any]] = None
    # Decomposition strategy fields
    sub_queries: List[str] = field(default_factory=list)
    decomposition_response: Optional[str] = None


# =============================================================================
# STEP-BACK PROMPTING
# =============================================================================

STEP_BACK_PROMPT = """You are a retrieval assistant. Given a user question, identify broader concepts
and principles that would help retrieve relevant information from a document collection.

The knowledge base covers: {corpus_topics}

TASK: Generate an effective search query that captures the underlying concepts.

PROCESS:
1. Identify the CORE TOPIC: What is the user fundamentally asking about?
2. Identify RELATED CONCEPTS: What foundational ideas, mechanisms, or principles relate?
3. Use CONCRETE VOCABULARY: Include specific terms likely to appear in relevant documents
4. BROADEN appropriately: Step back from specific details to capture the general theme

EXAMPLES:
User: "Why do I feel anxious before presentations?"
Query: "anxiety fear response public speaking stress performance social evaluation"

User: "How do I make better decisions?"
Query: "decision making cognitive processes judgment reasoning choice evaluation"

User: "What causes procrastination?"
Query: "procrastination delay motivation self-control temporal discounting willpower"

Generate ONLY the search query. Use 10-20 words of relevant concepts and terms."""


# =============================================================================
# MULTI-QUERY PROMPTS
# =============================================================================

PRINCIPLE_EXTRACTION_PROMPT = """You are analyzing a question for a knowledge retrieval system.

The knowledge base covers: {corpus_topics}

Given this question, extract the KEY UNDERLYING CONCEPTS that would help retrieve relevant passages from this document collection.

Question: "{query}"

Identify:
1. CORE TOPIC: The fundamental subject in 3-5 words
2. PRIMARY CONCEPTS: Specific mechanisms, theories, or frameworks (2-3 items)
3. SECONDARY CONCEPTS: Related ideas, schools of thought, or approaches (2-3 items)
4. RELATED TERMS: Vocabulary likely to appear in relevant passages (3-5 items)

EXAMPLES:

Question: "Why do we procrastinate?"
{{
  "core_topic": "procrastination and self-control",
  "primary_concepts": ["temporal discounting", "motivation", "willpower"],
  "secondary_concepts": ["self-regulation", "delay of gratification"],
  "related_terms": ["avoidance", "task aversion", "impulsivity", "planning"]
}}

Question: "How should I deal with anxiety?"
{{
  "core_topic": "anxiety management",
  "primary_concepts": ["stress response", "fear regulation", "coping mechanisms"],
  "secondary_concepts": ["cognitive reframing", "acceptance"],
  "related_terms": ["worry", "calm", "relaxation", "mindfulness", "exposure"]
}}

Now extract concepts for:
Question: "{query}"

Respond with JSON only."""


MULTI_QUERY_PROMPT = """Generate targeted search queries for a document retrieval system.

Original question: "{query}"

Extracted concepts:
- Core topic: {core_topic}
- Primary concepts: {primary_concepts}
- Secondary concepts: {secondary_concepts}
- Related terms: {related_terms}

Generate 4 diverse search queries (8-15 words each) to retrieve relevant passages:

1. TECHNICAL: Use specific mechanisms, processes, or terminology from the primary concepts
2. CONCEPTUAL: Use frameworks, theories, or abstract ideas from the secondary concepts
3. APPLIED: Focus on practical applications, examples, or real-world scenarios
4. BROAD: Use the core topic with accessible vocabulary

Respond with JSON:
{{
  "queries": [
    {{"type": "technical", "query": "..."}},
    {{"type": "conceptual", "query": "..."}},
    {{"type": "applied", "query": "..."}},
    {{"type": "broad", "query": "..."}}
  ]
}}"""


# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_PROMPT = """You break down complex questions into simpler sub-questions for a knowledge retrieval system.

The knowledge base covers: {corpus_topics}

TASK: Decompose this complex question into 2-4 simpler sub-questions that can be answered independently from this document collection.

RULES:
1. Each sub-question should be self-contained and answerable from the document collection
2. Include a synthesis question if the original asks for comparison or integration
3. Use specific terminology relevant to the topic
4. Keep sub-questions focused (not too broad)

EXAMPLES:

Question: "Compare different approaches to managing stress"
Sub-questions:
1. "What are the main psychological techniques for stress management?"
2. "What are the physiological approaches to reducing stress?"
3. "How do psychological and physiological stress management methods complement each other?"

Question: "How do experts explain the causes of procrastination?"
Sub-questions:
1. "What cognitive factors contribute to procrastination?"
2. "What emotional and motivational factors lead to procrastination?"
3. "What strategies are recommended to overcome procrastination?"

Question: "What are the benefits and drawbacks of remote work?"
Sub-questions:
1. "What are the productivity benefits of remote work?"
2. "What challenges do remote workers face?"
3. "How can organizations balance remote work benefits with its challenges?"

Now decompose this question:
Question: "{query}"

Respond with JSON:
{{
  "sub_questions": ["...", "...", "..."],
  "reasoning": "Brief explanation of decomposition"
}}"""


def step_back_prompt(query: str, model: Optional[str] = None) -> str:
    """Transform any query into a broader, more retrievable form.

    Step-back prompting abstracts specific questions into underlying concepts,
    improving retrieval by using vocabulary that matches the knowledge base.

    Args:
        query: The original open-ended query.
        model: Override model (defaults to PREPROCESSING_MODEL from config).

    Returns:
        A broader query that captures the underlying concepts.

    Example:
        >>> step_back_prompt("How should I live my life?")
        "Stoic and philosophical principles for living a good life"
    """
    model = model or PREPROCESSING_MODEL

    # Format prompt with corpus topics for vocabulary grounding
    system_prompt = STEP_BACK_PROMPT.format(corpus_topics=CORPUS_TOPICS)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,  # Slight creativity for better abstraction
            max_tokens=50,
        )

        return response.strip().strip('"')

    except requests.RequestException as e:
        logger.warning(f"Step-back prompt failed: {e}, using original query")
        return query


# =============================================================================
# MULTI-QUERY FUNCTIONS
# =============================================================================


def extract_principles(query: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Extract underlying principles and concepts from a query.

    This is the first step of multi-query generation, identifying the
    core concepts that should inform query generation. Uses Pydantic
    PrincipleExtraction schema for type-safe extraction.

    Args:
        query: The user's original query.
        model: Override model (defaults to PREPROCESSING_MODEL).

    Returns:
        Dictionary with core_topic, neuroscience_concepts,
        philosophical_concepts, and related_terms.

    Example:
        >>> result = extract_principles("Why do we need approval?")
        >>> result["core_topic"]
        "social validation and emotional reward"
    """
    model = model or PREPROCESSING_MODEL

    # Format prompt with corpus topics for vocabulary grounding
    prompt = PRINCIPLE_EXTRACTION_PROMPT.format(
        query=query,
        corpus_topics=CORPUS_TOPICS,
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_structured_completion(
            messages=messages,
            model=model,
            response_model=PrincipleExtraction,
            temperature=0.0,
            max_tokens=300,
        )

        # Convert Pydantic model to dict for backward compatibility
        return result.model_dump()

    except (PydanticValidationError, Exception) as e:
        logger.warning(f"Principle extraction error: {e}")
        # Return minimal default
        return {
            "core_topic": query,
            "primary_concepts": [],
            "secondary_concepts": [],
            "related_terms": [],
        }


def generate_multi_queries(
    query: str,
    principles: Dict[str, Any],
    model: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generate multiple targeted search queries from extracted principles.

    Creates 4 diverse queries targeting different aspects of the knowledge base:
    - technical: Specific mechanisms, processes, or terminology
    - conceptual: Frameworks, theories, or abstract ideas
    - applied: Practical applications or real-world scenarios
    - broad: Core topic in accessible language

    Uses Pydantic MultiQueryResult schema for type-safe extraction.

    Args:
        query: Original user query.
        principles: Output from extract_principles().
        model: Override model.

    Returns:
        List of dicts with 'type' and 'query' keys.

    Example:
        >>> principles = extract_principles("Why do we procrastinate?")
        >>> queries = generate_multi_queries("Why do we procrastinate?", principles)
        >>> len(queries)
        4
    """
    model = model or PREPROCESSING_MODEL

    prompt = MULTI_QUERY_PROMPT.format(
        query=query,
        core_topic=principles.get("core_topic", query),
        primary_concepts=", ".join(principles.get("primary_concepts", [])),
        secondary_concepts=", ".join(principles.get("secondary_concepts", [])),
        related_terms=", ".join(principles.get("related_terms", [])),
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_structured_completion(
            messages=messages,
            model=model,
            response_model=MultiQueryResult,
            temperature=0.3,  # Slight creativity for query variation
            max_tokens=400,
        )

        # Convert Pydantic models to dicts for backward compatibility
        return [q.model_dump() for q in result.queries]

    except (PydanticValidationError, Exception) as e:
        logger.warning(f"Multi-query generation error: {e}")
        # Fallback: return original query as single query
        return [{"type": "original", "query": query}]


# =============================================================================
# QUERY DECOMPOSITION
# =============================================================================


def decompose_query(query: str, model: Optional[str] = None) -> tuple[List[str], str]:
    """Decompose a complex query into sub-questions for parallel retrieval.

    Breaks complex comparison or multi-aspect questions into simpler
    sub-questions that can be answered independently. Each sub-question
    is used for retrieval, with results merged using RRF.

    Uses Pydantic DecompositionResult schema for type-safe extraction.
    The schema guarantees sub_questions is a List[str].

    Args:
        query: The user's original query.
        model: Override model (defaults to PREPROCESSING_MODEL).

    Returns:
        Tuple of (list of sub-questions, raw LLM response string).

    Example:
        >>> sub_qs, resp = decompose_query("Compare Stoic and Buddhist views on suffering")
        >>> len(sub_qs)
        3
        >>> sub_qs[0]
        "What is the Stoic view on suffering and how to overcome it?"
    """
    model = model or PREPROCESSING_MODEL

    # Format prompt with corpus topics for vocabulary grounding
    prompt = DECOMPOSITION_PROMPT.format(
        query=query,
        corpus_topics=CORPUS_TOPICS,
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_structured_completion(
            messages=messages,
            model=model,
            response_model=DecompositionResult,
            temperature=0.3,  # Slight creativity for varied sub-questions
            max_tokens=400,
        )

        # Pydantic guarantees sub_questions is List[str]
        if not result.sub_questions:
            logger.warning("[decompose] No sub-questions extracted, using original")
            return [query], result.model_dump_json()

        logger.info(f"[decompose] Generated {len(result.sub_questions)} sub-questions")
        return result.sub_questions, result.model_dump_json()

    except (PydanticValidationError, Exception) as e:
        logger.warning(f"[decompose] Error: {e}, using original query")
        return [query], str(e)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def preprocess_query(
    query: str,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
) -> PreprocessedQuery:
    """Preprocess a query for optimal retrieval.

    Main entry point for query preprocessing. Dispatches to the appropriate
    strategy based on the strategy parameter. Each strategy applies its
    transformation directly to the query without classification.

    Args:
        query: The user's original query.
        model: Override model for LLM calls.
        strategy: Preprocessing strategy ID. Options:
            - "none": Return original query unchanged (no LLM calls)
            - "step_back": Transform to broader concepts for better retrieval
            - "multi_query": Generate 4 targeted queries + RRF merge
            - "decomposition": Break into sub-questions + RRF merge

    Returns:
        PreprocessedQuery with transformed query and strategy_used.

    Example:
        >>> result = preprocess_query("How should I live?", strategy="step_back")
        >>> result.search_query
        "Stoic philosophy meaning purpose good life virtue wisdom"
        >>> result.strategy_used
        "step_back"
    """
    # Import here to avoid circular imports
    from src.config import DEFAULT_PREPROCESSING_STRATEGY
    from src.rag_pipeline.retrieval.preprocessing.strategies import get_strategy

    # Use default strategy if none specified
    if strategy is None:
        strategy = DEFAULT_PREPROCESSING_STRATEGY

    # Get and execute the strategy function
    strategy_fn = get_strategy(strategy)
    return strategy_fn(query, model=model)
