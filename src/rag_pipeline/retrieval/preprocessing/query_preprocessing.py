"""Query preprocessing for improved RAG retrieval.

## RAG Theory: Query Preprocessing

Query preprocessing improves retrieval by transforming queries before searching.
Each preprocessing strategy applies its transformation directly to any query,
following the original research papers' design.

Techniques implemented:
1. **HyDE** generates hypothetical answers for semantic matching
   - Paper: "Precise Zero-Shot Dense Retrieval" (arXiv:2212.10496)
2. **Query decomposition** breaks complex questions into sub-queries
   - Research: Query Decomposition showed +36.7% MRR@10 improvement (arXiv:2507.00355)

## Library Usage

Uses OpenRouter API via `requests` for LLM calls.
Pydantic schemas ensure structured LLM outputs.

## Data Flow

1. User selects preprocessing strategy (none, hyde, decomposition)
2. Strategy transforms query directly (no classification routing)
3. Return PreprocessedQuery with original, transformed, and metadata
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

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
        hyde_passage: The hypothetical passage generated (hyde strategy).
        sub_queries: Decomposed sub-questions (decomposition strategy).
        generated_queries: List of {type, query} dicts for multi-query retrieval.
        decomposition_response: Raw decomposition response from LLM.
    """

    original_query: str
    search_query: str
    strategy_used: str = ""
    preprocessing_time_ms: float = 0.0
    model: str = ""
    # HyDE strategy fields
    hyde_passage: Optional[str] = None
    # Decomposition strategy fields (generated_queries used for RRF)
    generated_queries: List[Dict[str, str]] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    decomposition_response: Optional[str] = None


# =============================================================================
# HyDE: HYPOTHETICAL DOCUMENT EMBEDDINGS
# =============================================================================

HYDE_PROMPT = """You are a knowledgeable assistant for a knowledge base covering:
{corpus_topics}

Given a question, write a SHORT hypothetical passage (2-3 sentences) that would answer it.
The passage should:
1. Sound like an excerpt from an encyclopedia or textbook
2. Include terminology from BOTH neuroscience and practical philosophy when related or relevant
3. Be factually plausible (even if you're not certain it's correct)

CROSS-DOMAIN EXAMPLES:

Question: "Why do we procrastinate?"
Passage: "Procrastination stems from temporal discounting, where the brain's limbic system overvalues immediate rewards over future goals. The prefrontal cortex struggles to maintain long-term focus. Stoic philosophers addressed this through practices of premeditation and focusing on what is within our control."

Question: "How can I control my impulses?"
Passage: "Impulse control involves the prefrontal cortex inhibiting limbic system responses, particularly the amygdala and reward circuits. Cognitive reappraisal engages dorsolateral PFC to reframe triggers. Stoic philosophy teaches distinguishing between what is 'up to us' (our judgments) and what is not (external events)."

Question: "What makes us happy?"
Passage: "Happiness involves both hedonic pleasure via the mesolimbic dopamine system and eudaimonic flourishing through meaning and virtue. The brain's reward circuits drive momentary satisfaction, while philosophical traditions from Aristotle to the Stoics emphasize lasting contentment through character development."

Now write a passage for:
Question: "{query}"

Passage:"""


# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_PROMPT = """You break down complex questions for a cross-domain knowledge retrieval system.

The knowledge base covers:
{corpus_topics}

DOMAIN STRUCTURE:
- NEUROSCIENCE domain: Explains HOW (brain mechanisms, biological processes)
- PRACTICAL PHILOSOPHY domain: Explains WHY and WHAT TO DO (wisdom, virtues, practices)

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


def hyde_prompt(query: str, model: Optional[str] = None) -> str:
    """Generate hypothetical answer for HyDE retrieval.

    HyDE (Hypothetical Document Embeddings) generates a plausible answer
    to the query, then searches for real passages similar to this answer.
    This bridges the semantic gap between questions and document passages.

    Paper: arXiv:2212.10496 - "Precise Zero-Shot Dense Retrieval without
    Relevance Labels"

    Args:
        query: The user's original question.
        model: Override model (defaults to PREPROCESSING_MODEL).

    Returns:
        A hypothetical passage that would answer the query.

    Example:
        >>> hyde_prompt("Why do we procrastinate?")
        "Procrastination stems from temporal discounting..."
    """
    model = model or PREPROCESSING_MODEL

    # Format prompt with corpus topics and query
    prompt = HYDE_PROMPT.format(
        corpus_topics=CORPUS_TOPICS,
        query=query,
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.7,  # Higher creativity for diverse answers
            max_tokens=150,   # Longer for 2-3 sentence passages
        )

        return response.strip()

    except requests.RequestException as e:
        logger.warning(f"HyDE prompt failed: {e}, using original query")
        return query


# =============================================================================
# QUERY DECOMPOSITION
# =============================================================================


def decompose_query(query: str, model: Optional[str] = None) -> Tuple[List[str], str]:
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
            - "hyde": Generate hypothetical answer for semantic matching
            - "decomposition": Break into sub-questions + RRF merge

    Returns:
        PreprocessedQuery with transformed query and strategy_used.

    Example:
        >>> result = preprocess_query("How should I live?", strategy="hyde")
        >>> result.search_query
        "Eudaimonia, the ancient Greek concept of flourishing..."
        >>> result.strategy_used
        "hyde"
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
