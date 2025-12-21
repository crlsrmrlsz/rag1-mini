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

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

import requests

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PREPROCESSING_MODEL,
)
from src.utils.file_utils import setup_logging

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
    """

    original_query: str
    query_type: QueryType
    search_query: str
    step_back_query: Optional[str] = None
    sub_queries: List[str] = field(default_factory=list)
    strategy_used: str = ""  # Strategy ID that was applied (none, baseline, step_back)
    preprocessing_time_ms: float = 0.0
    model: str = ""  # Model ID used for preprocessing
    classification_prompt_used: Optional[str] = None
    step_back_prompt_used: Optional[str] = None
    classification_response: Optional[str] = None  # Raw JSON from classification LLM
    step_back_response: Optional[str] = None  # Raw step-back query from LLM


# =============================================================================
# LLM API CALLS
# =============================================================================


def _call_chat_completion(
    messages: List[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    json_mode: bool = False,
) -> str:
    """Call OpenRouter chat completion API.

    Args:
        messages: List of message dicts with role and content.
        model: Model ID (e.g., "openai/gpt-5-nano").
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens to generate.
        json_mode: If True, request JSON response format.

    Returns:
        The assistant's response text.

    Raises:
        requests.RequestException: On API errors after retries.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    max_retries = 3
    backoff_base = 1.5

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

            if response.status_code >= 500 or response.status_code == 429:
                if attempt < max_retries:
                    delay = backoff_base ** (attempt + 1)
                    logger.warning(
                        f"API error {response.status_code}, retry {attempt + 1} after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

            response.raise_for_status()

        except requests.RequestException as exc:
            if attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(f"Request failed ({exc}), retry {attempt + 1} in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise

    raise requests.RequestException("Max retries exceeded")


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

    Uses an LLM to analyze the query intent and classify it appropriately.
    This classification determines which preprocessing strategy to apply.

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

    try:
        response = _call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=50,
            json_mode=True,
        )

        result = json.loads(response)
        query_type_str = result.get("query_type", "factual").lower()

        type_map = {
            "factual": QueryType.FACTUAL,
            "open_ended": QueryType.OPEN_ENDED,
            "multi_hop": QueryType.MULTI_HOP,
        }

        return type_map.get(query_type_str, QueryType.FACTUAL), response

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Classification parse error: {e}, defaulting to FACTUAL")
        return QueryType.FACTUAL, str(e)


# =============================================================================
# STEP-BACK PROMPTING
# =============================================================================

STEP_BACK_PROMPT = """You transform questions into effective search queries for a knowledge system about human nature.

KNOWLEDGE BASE CONTENTS:
- Neuroscience books: brain mechanisms, dopamine/serotonin/oxytocin, prefrontal cortex, amygdala, limbic system, decision-making, emotions, consciousness, evolutionary psychology
- Philosophy books: Stoicism (Marcus Aurelius, Epictetus, Seneca), Taoism (Lao Tzu, Chuang Tzu), Buddhism, virtue ethics, meaning of life, wisdom traditions

TASK: Generate a search query that will retrieve the most relevant passages.

PROCESS:
1. Identify the CORE TOPIC: What is the user really asking about? (e.g., fear, purpose, social needs, self-control)
2. Identify SPECIFIC MECHANISMS: What brain systems, psychological processes, or philosophical concepts relate?
3. Use CONCRETE VOCABULARY: Include specific terms from the knowledge base (author names, brain regions, philosophical schools, emotions)

EXAMPLES:
User: "Why do I feel anxious?"
Think: Core=anxiety/fear, Mechanisms=amygdala+cortisol+fight-or-flight+Stoic tranquility
Query: "amygdala fear response anxiety Stoic tranquility ataraxia Epictetus control"

User: "Why do we need approval from others to feel good?"
Think: Core=social validation+reward, Mechanisms=dopamine+social brain+oxytocin+Stoic indifference to externals
Query: "dopamine social reward approval seeking Stoic virtue external validation Marcus Aurelius"

User: "What is the point of life?"
Think: Core=meaning/purpose, Mechanisms=prefrontal cortex goal-setting+existential psychology+Stoic eudaimonia+Taoist wu-wei
Query: "meaning purpose life eudaimonia Stoicism Taoism Viktor Frankl prefrontal goals"

User: "How can I control my anger?"
Think: Core=anger regulation, Mechanisms=amygdala+prefrontal inhibition+Seneca on anger+cognitive reappraisal
Query: "anger regulation amygdala prefrontal Seneca De Ira Stoic passion cognitive reappraisal"

Generate ONLY the search query. Use 10-20 words. Include both neuroscience and philosophy terms."""


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
        response = _call_chat_completion(
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
# MAIN ENTRY POINT
# =============================================================================


def preprocess_query(
    query: str,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
    enable_step_back: bool = True,  # DEPRECATED: Use strategy parameter instead
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
        enable_step_back: DEPRECATED. Use strategy="baseline" to skip step-back.
            Kept for backward compatibility with existing callers.

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
    from src.preprocessing.strategies import get_strategy

    # Handle strategy selection with backward compatibility
    if strategy is None:
        if enable_step_back:
            strategy = DEFAULT_PREPROCESSING_STRATEGY
        else:
            # enable_step_back=False maps to baseline (classify only)
            strategy = "baseline"

    # Get and execute the strategy function
    strategy_fn = get_strategy(strategy)
    return strategy_fn(query, model=model)
