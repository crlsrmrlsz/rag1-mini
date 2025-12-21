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
        preprocessing_time_ms: Time taken for preprocessing in milliseconds.
        classification_prompt_used: The prompt sent to LLM for classification (for logging).
        step_back_prompt_used: The prompt sent to LLM for step-back (for logging).
    """

    original_query: str
    query_type: QueryType
    search_query: str
    step_back_query: Optional[str] = None
    sub_queries: List[str] = field(default_factory=list)
    preprocessing_time_ms: float = 0.0
    classification_prompt_used: Optional[str] = None
    step_back_prompt_used: Optional[str] = None


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

CLASSIFICATION_PROMPT = """You are a query classifier for a knowledge system about human behavior, mind, and life.

The system contains:
- Neuroscience texts: brain mechanisms, consciousness, emotions, decision-making, behavior
- Philosophy texts: Stoicism, Taoism, wisdom traditions, meaning, ethics, the good life

Classify the query based on what type of response would best serve the user:

1. "factual" - Seeks a specific fact, definition, mechanism, or what a particular author said
   -> Direct, precise answer needed

2. "open_ended" - Explores human nature, behavior, emotions, life questions, or seeks understanding
   -> Rich, multi-perspective answer integrating science and wisdom where relevant

3. "multi_hop" - Explicitly compares, contrasts, or connects concepts across sources or domains
   -> Analytical synthesis across perspectives

When uncertain, prefer "open_ended" - most questions about human behavior benefit from integrated perspectives.

Respond with JSON: {"query_type": "factual" | "open_ended" | "multi_hop"}"""


def classify_query(query: str, model: Optional[str] = None) -> QueryType:
    """Classify a query into FACTUAL, OPEN_ENDED, or MULTI_HOP.

    Uses an LLM to analyze the query intent and classify it appropriately.
    This classification determines which preprocessing strategy to apply.

    Args:
        query: The user's search query.
        model: Override model (defaults to PREPROCESSING_MODEL from config).

    Returns:
        QueryType enum value.

    Example:
        >>> classify_query("What is serotonin?")
        QueryType.FACTUAL
        >>> classify_query("How should I live my life?")
        QueryType.OPEN_ENDED
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

        return type_map.get(query_type_str, QueryType.FACTUAL)

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Classification parse error: {e}, defaulting to FACTUAL")
        return QueryType.FACTUAL


# =============================================================================
# STEP-BACK PROMPTING
# =============================================================================

STEP_BACK_PROMPT = """You help improve search queries for a knowledge system about human behavior.

The system contains neuroscience (brain, behavior, consciousness) AND philosophy (wisdom, ethics, meaning).

Given the user's question, generate a broader "step-back" query that will retrieve relevant content
from both scientific AND philosophical perspectives when appropriate.

Principles:
- Identify the UNDERLYING TOPIC (emotion, motivation, decision-making, meaning, consciousness, etc.)
- Include SCIENTIFIC ANGLE (brain mechanisms, neurotransmitters, psychology, evolution)
- Include WISDOM ANGLE (philosophical practices, ancient wisdom, life guidance)
- Keep it broad enough to catch diverse relevant passages

Examples of the pattern:
- Specific question -> "underlying topic from neuroscience; underlying topic from philosophy"
- "Why do I feel anxious?" -> "neuroscience of anxiety and fear; philosophical approaches to tranquility"
- "What is the point of life?" -> "psychology of meaning and purpose; philosophical perspectives on the good life"

Generate ONLY the step-back query. Keep it under 20 words."""


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
    enable_step_back: bool = True,
) -> PreprocessedQuery:
    """Preprocess a query for optimal retrieval.

    Main entry point for query preprocessing. Classifies the query and applies
    appropriate transformations based on query type.

    Args:
        query: The user's original query.
        model: Override model for LLM calls.
        enable_step_back: If True, apply step-back prompting for open-ended queries.

    Returns:
        PreprocessedQuery with classification and transformed query.

    Example:
        >>> result = preprocess_query("How should I live?")
        >>> result.query_type
        QueryType.OPEN_ENDED
        >>> result.search_query
        "Stoic and philosophical principles for living a good life"
    """
    start_time = time.time()

    # Step 1: Classify the query
    query_type = classify_query(query, model=model)
    logger.info(f"Query classified as: {query_type.value}")

    # Step 2: Apply appropriate transformation
    step_back_query = None
    search_query = query
    step_back_prompt_used = None

    if query_type == QueryType.OPEN_ENDED and enable_step_back:
        step_back_query = step_back_prompt(query, model=model)
        search_query = step_back_query
        step_back_prompt_used = STEP_BACK_PROMPT
        logger.info(f"Step-back query: {step_back_query}")

    elif query_type == QueryType.MULTI_HOP:
        # Future: Implement query decomposition
        # For now, use original query
        logger.info("Multi-hop query detected (decomposition not yet implemented)")

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        query_type=query_type,
        search_query=search_query,
        step_back_query=step_back_query,
        preprocessing_time_ms=elapsed_ms,
        classification_prompt_used=CLASSIFICATION_PROMPT,
        step_back_prompt_used=step_back_prompt_used,
    )
