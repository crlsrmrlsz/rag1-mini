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

HYDE_PROMPT = """You are generating hypothetical passages for a knowledge base covering:
{corpus_topics}

Given a question, write a SHORT hypothetical passage (2-3 sentences) that would answer it.

CRITICAL: Each passage must give EQUAL weight to:
- NEUROSCIENCE (50%): Brain mechanisms, cognitive processes, biological findings
- PHILOSOPHY (50%): Wisdom traditions (Stoics, Schopenhauer, Taoism, Confucius, Gracián)

BALANCED EXAMPLES:

Question: "Why do we procrastinate?"
Passage: "Procrastination involves temporal discounting in the brain's reward system—the limbic system overvalues immediate rewards while the prefrontal cortex struggles to maintain future goals. Seneca observed that we squander time because we treat our lifespan as infinite, failing to value each moment as the precious resource it is."

Question: "How can I control my impulses?"
Passage: "The prefrontal cortex must override automatic amygdala-driven responses, a process requiring cognitive effort and often depleted by stress or fatigue. Epictetus taught that only our judgments are truly 'up to us'—recognizing this distinction allows us to respond thoughtfully rather than react automatically."

Question: "What brings lasting contentment?"
Passage: "Neuroscience reveals that hedonic adaptation limits satisfaction from external rewards—the brain quickly returns to baseline after pleasurable experiences. Schopenhauer argued that lasting contentment comes not from acquiring more, but from the absence of unsatisfied desire and the cultivation of inner resources."

Question: "How do we make better decisions?"
Passage: "Kahneman's research shows System 1 (fast, intuitive) often produces biased judgments that System 2 (slow, deliberate) can correct with effort. Gracián counseled never deciding in passion's heat: 'reflection is the safeguard of prudence'—allowing emotions to settle before committing to action."

Question: "Why do we suffer?"
Passage: "Chronic stress activates the hypothalamic-pituitary-adrenal axis, flooding the body with cortisol and altering brain structure over time. Schopenhauer viewed suffering as inherent to existence itself—the will endlessly strives, and satisfaction is always temporary, giving way to new desires or boredom."

Question: "How should we respond to what we cannot control?"
Passage: "The brain's anterior cingulate cortex monitors conflicts between desire and reality, often triggering frustration when expectations aren't met. The Tao te ching teaches that the sage flows with circumstances like water—soft yet persistent, yielding yet ultimately shaping the landscape it encounters."

Now write a passage for:
Question: "{query}"

Passage:"""


# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_PROMPT = """You break down complex questions for a cross-domain knowledge retrieval system.

The knowledge base covers:
{corpus_topics}

TASK: Decompose this question into 3-4 sub-questions with EQUAL representation from neuroscience and philosophy.

VOCABULARY FROM THE KNOWLEDGE BASE:

NEUROSCIENCE - Use terms like:
- Decision/cognition: System 1/System 2, heuristics, cognitive biases, cognitive load, temporal discounting
- Brain regions: prefrontal cortex, amygdala, anterior cingulate cortex, limbic system, hippocampus
- Mechanisms: dopamine, serotonin, cortisol, HPA axis, hedonic adaptation, reward baseline
- Behavior: impulse control, cognitive reappraisal, stress response, fear conditioning

PHILOSOPHY - Use terms authentic to each tradition:
- Stoics: "what is up to us", impressions, assent, virtue as only good, tranquility, preferred indifferents
- Schopenhauer: will to live, denial of will, absence of desire, thing-in-itself, vanity of existence
- Taoism: wu wei (non-action), the Tao, soft overcomes hard, inner stillness, the sage
- Confucius: ren (benevolence), junzi (superior person), li (ritual propriety), filial piety
- Gracián: prudence, discretion, fortune, concealment, strategic self-presentation

BALANCED EXAMPLES:

Question: "How can I become less anxious?"
Sub-questions:
1. "How do the amygdala and prefrontal cortex interact during anxiety, and what calms the stress response?"
2. "What does Epictetus's 'dichotomy of control' teach about letting go of what is not 'up to us'?"
3. "What does the Tao te ching teach about yielding to circumstances rather than resisting them?"
4. "How do contemplative practices affect cortisol levels and brain stress circuits?"

Question: "Why is self-control so hard?"
Sub-questions:
1. "What causes prefrontal cortex fatigue and how does it lead to self-control failure?"
2. "What does Schopenhauer teach about the 'will to live' and why desire always returns?"
3. "What does Gracián advise about prudence and waiting until passion subsides before deciding?"
4. "How do dopamine reward systems drive impulsive behavior despite long-term goals?"

Question: "What makes people happy?"
Sub-questions:
1. "What does neuroscience reveal about hedonic adaptation and the brain's reward baseline?"
2. "What does Schopenhauer argue about contentment through 'absence of unsatisfied desire'?"
3. "What does Confucius teach about the happiness of cultivating ren (benevolence) and becoming a junzi?"
4. "How do Kahneman's findings on System 1/2 biases affect our pursuit of satisfaction?"

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
