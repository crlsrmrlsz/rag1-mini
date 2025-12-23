"""Query preprocessing for improved RAG retrieval.

## RAG Theory: Query Preprocessing

Query preprocessing improves retrieval by transforming queries before searching.
Each preprocessing strategy applies its transformation directly to any query,
following the original research papers' design.

Techniques implemented:
1. **Step-back prompting** abstracts questions into broader concepts
   - Paper: "Take a Step Back" (Google DeepMind, arXiv:2310.06117) showed +27% on multi-hop
2. **Query decomposition** breaks complex questions into sub-queries
   - Research: Query Decomposition showed +36.7% MRR@10 improvement

## Library Usage

Uses OpenRouter API via `requests` for LLM calls.
Pydantic schemas ensure structured LLM outputs.

## Data Flow

1. User selects preprocessing strategy (none, step_back, decomposition)
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
    # Decomposition strategy fields (generated_queries used for RRF)
    generated_queries: List[Dict[str, str]] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    decomposition_response: Optional[str] = None


# =============================================================================
# STEP-BACK PROMPTING
# =============================================================================

STEP_BACK_PROMPT = """You are a retrieval assistant for a knowledge base covering:
{corpus_topics}

This corpus spans TWO DOMAINS:
1. NEUROSCIENCE: Brain mechanisms, neural circuits, cognitive processes, behavioral biology
2. WISDOM TRADITIONS: Stoic, Taoist, Confucian, and Schopenhauerian philosophy

TASK: Transform the user's question into a search query that captures BOTH domains.

PROCESS:
1. Identify the CORE THEME: What is the user fundamentally asking about?
2. Extract MECHANISM concepts: How does the brain/biology handle this? (technical terms)
3. Extract PRINCIPLE concepts: What do wisdom traditions say? (philosophical terms)
4. Combine into a query using vocabulary from BOTH domains

CROSS-DOMAIN EXAMPLES:

User: "How can I control my impulses?"
Query: "self-control prefrontal cortex impulse regulation willpower Stoic desire mastery virtue temperance"

User: "Why do we fear death?"
Query: "mortality fear amygdala terror management anxiety Stoic death acceptance memento mori tranquility"

User: "How do I make better decisions?"
Query: "decision-making cognitive biases System 1 System 2 heuristics Stoic prudence practical wisdom deliberation"

User: "Why do we procrastinate?"
Query: "procrastination temporal discounting dopamine reward delay self-control Stoic present moment virtue action"

Generate ONLY the search query. Use 12-20 words spanning both neuroscience and philosophy vocabulary."""


# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_PROMPT = """You break down complex questions for a cross-domain knowledge retrieval system.

The knowledge base covers:
{corpus_topics}

DOMAIN STRUCTURE:
- NEUROSCIENCE domain: Explains HOW (brain mechanisms, biological processes)
- PHILOSOPHY domain: Explains WHY and WHAT TO DO (wisdom, virtues, practices)

TASK: Decompose this question into 3-4 sub-questions that target different aspects:

DECOMPOSITION STRATEGY:
1. First, identify if the question spans both domains (most do)
2. Create sub-questions that:
   - MECHANISM: Ask "how does the brain/body handle this?"
   - PRINCIPLE: Ask "what do wisdom traditions advise?"
   - SYNTHESIS: Ask "how do these perspectives connect?"

CROSS-DOMAIN EXAMPLES:

Question: "How can I become less anxious?"
Sub-questions:
1. "What brain mechanisms underlie anxiety and the stress response?"
2. "What do Stoic philosophers teach about managing worry and fear?"
3. "How do modern stress-reduction techniques relate to ancient wisdom practices?"

Question: "Why is self-control so hard?"
Sub-questions:
1. "How does the prefrontal cortex regulate impulses and what causes self-control failure?"
2. "What techniques do Stoics and Taoists recommend for mastering desires?"
3. "How do philosophical concepts of willpower compare to neuroscience findings?"

Question: "What makes people happy?"
Sub-questions:
1. "What does neuroscience reveal about reward systems and hedonic adaptation?"
2. "How do Stoic, Buddhist, and Schopenhauerian views define happiness and contentment?"
3. "Where do scientific and philosophical accounts of well-being converge?"

Now decompose:
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
