"""Query classification and preprocessing for improved RAG retrieval.

## RAG Theory: Query Preprocessing

Query preprocessing improves retrieval by transforming queries before searching.
This is based on research showing that query characteristics strongly affect
retrieval quality:

1. **Classification** identifies query type to apply appropriate strategy
2. **Step-back prompting** abstracts specific questions into broader concepts
   - Paper: "Take a Step Back" (Google DeepMind, 2023) showed +27% on multi-hop
   - Example: "What is the puppet metaphor?" -> "Stoic philosophy on emotions"
3. **Query decomposition** breaks complex questions into sub-queries
   - Paper: "Query Decomposition" showed +36.7% MRR@10 improvement

## Library Usage

Uses OpenRouter API via `requests` for LLM calls (same pattern as embed_texts.py).
JSON mode ensures structured classification output.

## Data Flow

1. User query enters preprocess_query()
2. classify_query() determines type via LLM
3. Based on type:
   - FACTUAL: Return original query (direct retrieval works well)
   - OPEN_ENDED: Apply step_back_prompt() to broaden query
   - MULTI_HOP: (Future) Apply query decomposition
4. Return PreprocessedQuery with original, transformed, and metadata
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

import requests
from pydantic import ValidationError as PydanticValidationError

from src.config import PREPROCESSING_MODEL
from src.shared.file_utils import setup_logging
from src.shared.openrouter_client import call_chat_completion, call_structured_completion
from src.rag_pipeline.retrieval.preprocessing.schemas import (
    ClassificationResult,
    PrincipleExtraction,
    MultiQueryResult,
    DecompositionResult,
)

logger = setup_logging(__name__)


class QueryType(Enum):
    """Classification of user queries for retrieval optimization.

    - FACTUAL: Seeks specific facts, definitions, or data points.
      Example: "What neurotransmitter is made from tryptophan?"

    - OPEN_ENDED: Philosophical, life-advice, or wisdom questions.
      Example: "How should I live my life?"

    - MULTI_HOP: Requires combining information from multiple sources.
      Example: "Compare Stoic and Taoist views on desire."
    """

    FACTUAL = "factual"
    OPEN_ENDED = "open_ended"
    MULTI_HOP = "multi_hop"


@dataclass
class PreprocessedQuery:
    """Result of query preprocessing.

    Attributes:
        original_query: The user's original query text.
        query_type: Classification result (FACTUAL, OPEN_ENDED, MULTI_HOP).
        search_query: The query to use for retrieval (may be transformed).
        step_back_query: For OPEN_ENDED, the abstracted broader query.
        sub_queries: For MULTI_HOP, decomposed sub-questions (future).
        strategy_used: The preprocessing strategy that was applied.
        preprocessing_time_ms: Time taken for preprocessing in milliseconds.
        model: Model ID used for preprocessing (for logging).
        classification_prompt_used: The prompt sent to LLM for classification (for logging).
        step_back_prompt_used: The prompt sent to LLM for step-back (for logging).
        classification_response: Raw JSON from classification LLM.
        step_back_response: Raw step-back query from LLM.
        generated_queries: List of {type, query} dicts for multi-query retrieval.
        principle_extraction: Raw principle extraction result for logging.
        principle_extraction_prompt_used: Prompt for principle extraction.
        multi_query_prompt_used: Prompt for multi-query generation.
    """

    original_query: str
    query_type: QueryType
    search_query: str
    step_back_query: Optional[str] = None
    sub_queries: List[str] = field(default_factory=list)
    strategy_used: str = ""  # Strategy ID that was applied
    preprocessing_time_ms: float = 0.0
    model: str = ""  # Model ID used for preprocessing
    classification_prompt_used: Optional[str] = None
    step_back_prompt_used: Optional[str] = None
    classification_response: Optional[str] = None
    step_back_response: Optional[str] = None
    # Multi-query fields
    generated_queries: List[Dict[str, str]] = field(default_factory=list)
    principle_extraction: Optional[Dict[str, Any]] = None
    principle_extraction_prompt_used: Optional[str] = None
    multi_query_prompt_used: Optional[str] = None
    # Decomposition fields (for MULTI_HOP)
    decomposition_prompt_used: Optional[str] = None
    decomposition_response: Optional[str] = None


# =============================================================================
# QUERY CLASSIFICATION
# =============================================================================

CLASSIFICATION_PROMPT = """You are a query classifier for a knowledge system about human nature.

The system contains:
- Neuroscience: brain mechanisms, emotions, decision-making, consciousness
- Philosophy: Stoicism, Taoism, wisdom traditions, meaning, ethics, the good life

Classify based on what response type would BEST serve the user:

## FACTUAL
Questions seeking specific, verifiable information:
- Definitions: "What is serotonin?"
- Specific mechanisms: "What brain region processes fear?"
- Author quotes: "What did Marcus Aurelius say about anger?"
- Technical details: "How many neurons are in the prefrontal cortex?"

## OPEN_ENDED
Questions about human nature, life, meaning, or behavior that benefit from multiple perspectives:
- "Why do we need approval from others to feel good?"
- "How can I deal with anxiety?"
- "What makes life meaningful?"
- "Why do humans fear death?"

Key signal: Questions asking "why do we/humans..." about emotions, behavior, or life are almost always open_ended.

## MULTI_HOP
Questions that EXPLICITLY compare or connect across domains:
- "Compare Stoic and Buddhist views on suffering"
- "How does neuroscience explain what the Stoics called 'passion'?"

## Decision Rule
If a question could benefit from BOTH scientific AND philosophical perspectives, classify as "open_ended".

Respond with JSON: {"query_type": "factual" | "open_ended" | "multi_hop"}"""


def classify_query(query: str, model: Optional[str] = None) -> tuple[QueryType, str]:
    """Classify a query into FACTUAL, OPEN_ENDED, or MULTI_HOP.

    Uses an LLM with Pydantic schema enforcement to analyze query intent.
    The ClassificationResult schema guarantees valid query_type values.

    Args:
        query: The user's search query.
        model: Override model (defaults to PREPROCESSING_MODEL from config).

    Returns:
        Tuple of (QueryType enum value, raw LLM response string).

    Example:
        >>> query_type, response = classify_query("What is serotonin?")
        >>> query_type
        QueryType.FACTUAL
        >>> response
        '{"query_type": "factual"}'
    """
    model = model or PREPROCESSING_MODEL

    messages = [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
        {"role": "user", "content": query},
    ]

    type_map = {
        "factual": QueryType.FACTUAL,
        "open_ended": QueryType.OPEN_ENDED,
        "multi_hop": QueryType.MULTI_HOP,
    }

    try:
        result = call_structured_completion(
            messages=messages,
            model=model,
            response_model=ClassificationResult,
            temperature=0.0,
            max_tokens=50,
        )

        # result.query_type is guaranteed to be one of the Literal values
        return type_map[result.query_type], result.model_dump_json()

    except (PydanticValidationError, Exception) as e:
        logger.warning(f"Classification error: {e}, defaulting to FACTUAL")
        return QueryType.FACTUAL, str(e)


# =============================================================================
# STEP-BACK PROMPTING
# =============================================================================

STEP_BACK_PROMPT = """You transform questions into effective search queries for a knowledge system about human nature.

KNOWLEDGE BASE CONTENTS:
- Neuroscience books: brain mechanisms, dopamine/serotonin/oxytocin, prefrontal cortex, amygdala, limbic system, decision-making, emotions, consciousness, evolutionary psychology
- Philosophy books:
  - Stoicism: Marcus Aurelius (Meditations), Seneca (Letters), Epictetus (Enchiridion, Art of Living)
  - Schopenhauer: pessimism, will, suffering, contentment, solitude
  - Taoism: Lao Tzu (Tao Te Ching), wu-wei, naturalness, balance
  - Confucianism: Confucius (Analects), virtue, ritual, relationships
  - Practical wisdom: Gracian (Art of Prudence), strategy, worldly wisdom
  - Behavioral psychology: Kahneman (Thinking Fast and Slow), cognitive biases, System 1/2

TASK: Generate a search query that will retrieve the most relevant passages.

PROCESS:
1. Identify the CORE TOPIC: What is the user really asking about? (e.g., fear, purpose, social needs, self-control)
2. Identify SPECIFIC MECHANISMS: What brain systems, psychological processes, or philosophical concepts relate?
3. Use CONCRETE VOCABULARY: Include specific terms from the knowledge base (author names, brain regions, philosophical schools, emotions)
4. BALANCE SOURCES: Include terms from MULTIPLE philosophical traditions, not just one

EXAMPLES:
User: "Why do I feel anxious?"
Think: Core=anxiety/fear, Mechanisms=amygdala+cortisol+fight-or-flight, Philosophy=Stoic tranquility+Schopenhauer suffering
Query: "amygdala fear response anxiety cortisol Stoic tranquility Schopenhauer suffering will"

User: "Why do we need approval from others?"
Think: Core=social validation, Mechanisms=dopamine+oxytocin+social brain, Philosophy=Confucian relationships+Kahneman biases
Query: "dopamine social reward approval Confucius relationships virtue Kahneman System 1 heuristics"

User: "What is the point of life?"
Think: Core=meaning/purpose, Mechanisms=prefrontal goals+reward system, Philosophy=Taoist wu-wei+Gracian prudence
Query: "meaning purpose prefrontal goals Lao Tzu wu-wei Tao naturalness Gracian prudence wisdom"

User: "How can I control my anger?"
Think: Core=anger regulation, Mechanisms=amygdala+prefrontal inhibition, Philosophy=Seneca De Ira+Schopenhauer will
Query: "anger regulation amygdala prefrontal Seneca De Ira Schopenhauer will cognitive reappraisal"

Generate ONLY the search query. Use 10-20 words. Include both neuroscience and philosophy terms."""


# =============================================================================
# MULTI-QUERY PROMPTS
# =============================================================================

PRINCIPLE_EXTRACTION_PROMPT = """You are analyzing a question about human nature for a knowledge retrieval system.

KNOWLEDGE BASE CONTENTS:
- Neuroscience books: brain mechanisms, neurotransmitters (dopamine, serotonin, oxytocin), brain regions (prefrontal cortex, amygdala, insula), emotions, decision-making, consciousness, evolutionary psychology
- Philosophy books:
  - Stoicism: Marcus Aurelius (Meditations), Seneca (Letters), Epictetus (Enchiridion, Art of Living)
  - Schopenhauer: pessimism, will, suffering, contentment, solitude
  - Taoism: Lao Tzu (Tao Te Ching), wu-wei, naturalness, balance
  - Confucianism: Confucius (Analects), virtue, ritual, relationships
  - Practical wisdom: Gracian (Art of Prudence), strategy, worldly wisdom
  - Behavioral psychology: Kahneman (Thinking Fast and Slow), cognitive biases, System 1/2

Given this question, extract the KEY UNDERLYING CONCEPTS that would help retrieve relevant passages:

Question: "{query}"

Identify:
1. CORE TOPIC: What is the fundamental subject? (e.g., "social reward", "anxiety", "purpose")
2. NEUROSCIENCE CONCEPTS: Specific mechanisms, brain regions, or processes (2-3 items)
3. PHILOSOPHICAL CONCEPTS: Relevant schools, authors, or ideas (2-3 items)
4. RELATED TERMS: Vocabulary likely to appear in relevant passages (3-5 items)

Respond with JSON:
{{
  "core_topic": "...",
  "neuroscience_concepts": ["...", "..."],
  "philosophical_concepts": ["...", "..."],
  "related_terms": ["...", "..."]
}}"""


MULTI_QUERY_PROMPT = """Generate targeted search queries for a hybrid neuroscience + philosophy knowledge base.

Original question: "{query}"

Extracted concepts:
- Core topic: {core_topic}
- Neuroscience: {neuro_concepts}
- Philosophy: {philo_concepts}
- Related terms: {related_terms}

Generate 4 search queries that will retrieve diverse, relevant passages:

1. NEUROSCIENCE query: Use specific brain regions, neurotransmitters, psychological mechanisms
2. PHILOSOPHY query: Use specific traditions, authors, philosophical concepts
3. BRIDGING query: Connect scientific and philosophical perspectives
4. BROAD query: Use the core topic in accessible language

Each query should be 8-15 words. Mix conceptual phrases with specific vocabulary.

Respond with JSON:
{{
  "queries": [
    {{"type": "neuroscience", "query": "..."}},
    {{"type": "philosophy", "query": "..."}},
    {{"type": "bridging", "query": "..."}},
    {{"type": "broad", "query": "..."}}
  ]
}}"""


# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_PROMPT = """You break down complex questions into simpler sub-questions for a knowledge retrieval system.

The knowledge base contains:
- NEUROSCIENCE: Brain mechanisms, neurotransmitters, emotions, decision-making, consciousness
- PHILOSOPHY:
  - Stoicism: Marcus Aurelius, Seneca, Epictetus
  - Schopenhauer: will, suffering, contentment
  - Taoism: Lao Tzu, wu-wei, naturalness
  - Confucianism: Confucius, virtue, relationships
  - Gracian: practical wisdom, prudence
  - Kahneman: cognitive biases, System 1/2

TASK: Decompose this complex question into 2-4 simpler sub-questions that can be answered independently.

RULES:
1. Each sub-question should be self-contained and answerable from a single domain
2. Include a synthesis question if the original asks for comparison or integration
3. Use specific terminology from the knowledge base
4. Keep sub-questions focused (not too broad)

EXAMPLES:

Question: "Compare Stoic and Schopenhauer's approaches to suffering"
Sub-questions:
1. "What is the Stoic view on suffering and how to overcome it?"
2. "What is Schopenhauer's teaching on suffering and the will?"
3. "How do Stoic and Schopenhauer approaches to suffering differ?"

Question: "How does neuroscience explain what philosophers call akrasia?"
Sub-questions:
1. "What is akrasia in philosophy and which philosophers discussed it?"
2. "What brain mechanisms are involved in self-control failures?"
3. "How do prefrontal-limbic interactions relate to weakness of will?"

Question: "What do both science and ancient wisdom say about anger management?"
Sub-questions:
1. "What brain mechanisms underlie anger and its regulation?"
2. "What did Seneca and the Stoics teach about managing anger?"
3. "How do modern psychology findings align with ancient anger management wisdom?"

Now decompose this question:
Question: "{query}"

Respond with JSON:
{{
  "sub_questions": ["...", "...", "..."],
  "reasoning": "Brief explanation of decomposition"
}}"""


def step_back_prompt(query: str, model: Optional[str] = None) -> str:
    """Transform an open-ended query into a broader, more retrievable form.

    Step-back prompting abstracts specific questions into underlying principles,
    improving retrieval for philosophical and wisdom-seeking queries.

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

    messages = [
        {"role": "system", "content": STEP_BACK_PROMPT},
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

    prompt = PRINCIPLE_EXTRACTION_PROMPT.format(query=query)

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
            "neuroscience_concepts": [],
            "philosophical_concepts": [],
            "related_terms": [],
        }


def generate_multi_queries(
    query: str,
    principles: Dict[str, Any],
    model: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generate multiple targeted search queries from extracted principles.

    Creates 4 diverse queries targeting different aspects of the knowledge base:
    - neuroscience: Brain regions, neurotransmitters, mechanisms
    - philosophy: Traditions, authors, concepts
    - bridging: Connecting scientific and philosophical perspectives
    - broad: Core topic in accessible language

    Uses Pydantic MultiQueryResult schema for type-safe extraction.

    Args:
        query: Original user query.
        principles: Output from extract_principles().
        model: Override model.

    Returns:
        List of dicts with 'type' and 'query' keys.

    Example:
        >>> principles = extract_principles("Why do we fear death?")
        >>> queries = generate_multi_queries("Why do we fear death?", principles)
        >>> len(queries)
        4
    """
    model = model or PREPROCESSING_MODEL

    prompt = MULTI_QUERY_PROMPT.format(
        query=query,
        core_topic=principles.get("core_topic", query),
        neuro_concepts=", ".join(principles.get("neuroscience_concepts", [])),
        philo_concepts=", ".join(principles.get("philosophical_concepts", [])),
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
    """Decompose a MULTI_HOP query into sub-questions.

    Breaks complex comparison or multi-aspect questions into simpler
    sub-questions that can be answered independently. Each sub-question
    is then used for retrieval, with results merged using RRF.

    Uses Pydantic DecompositionResult schema for type-safe extraction.
    The schema guarantees sub_questions is a List[str].

    Args:
        query: The user's original query (should be MULTI_HOP type).
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
    prompt = DECOMPOSITION_PROMPT.format(query=query)

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
    strategy based on the strategy parameter.

    Args:
        query: The user's original query.
        model: Override model for LLM calls.
        strategy: Preprocessing strategy ID. Options:
            - "none": Return original query unchanged (no LLM calls)
            - "baseline": Classify only, no transformation
            - "step_back": Classify + step-back for open-ended (default)

    Returns:
        PreprocessedQuery with classification, transformed query, and strategy_used.

    Example:
        >>> result = preprocess_query("How should I live?", strategy="step_back")
        >>> result.query_type
        QueryType.OPEN_ENDED
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
