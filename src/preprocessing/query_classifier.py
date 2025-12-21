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
from typing import Optional, List, Dict, Any

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


# =============================================================================
# MULTI-QUERY PROMPTS
# =============================================================================

PRINCIPLE_EXTRACTION_PROMPT = """You are analyzing a question about human nature for a knowledge retrieval system.

KNOWLEDGE BASE CONTENTS:
- Neuroscience books: brain mechanisms, neurotransmitters (dopamine, serotonin, oxytocin), brain regions (prefrontal cortex, amygdala, insula), emotions, decision-making, consciousness, evolutionary psychology
- Philosophy books: Stoicism (Marcus Aurelius, Epictetus, Seneca), Taoism (Lao Tzu, Chuang Tzu), Buddhism, wisdom traditions, virtue ethics, meaning of life

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
- PHILOSOPHY: Stoicism (Marcus Aurelius, Epictetus, Seneca), Taoism, Buddhism, virtue ethics, wisdom traditions

TASK: Decompose this complex question into 2-4 simpler sub-questions that can be answered independently.

RULES:
1. Each sub-question should be self-contained and answerable from a single domain
2. Include a synthesis question if the original asks for comparison or integration
3. Use specific terminology from the knowledge base
4. Keep sub-questions focused (not too broad)

EXAMPLES:

Question: "Compare Stoic and Buddhist approaches to suffering"
Sub-questions:
1. "What is the Stoic view on suffering and how to overcome it?"
2. "What is the Buddhist teaching on suffering and its cessation?"
3. "How do Stoic and Buddhist approaches to suffering differ?"

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
# MULTI-QUERY FUNCTIONS
# =============================================================================


def extract_principles(query: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Extract underlying principles and concepts from a query.

    This is the first step of multi-query generation, identifying the
    core concepts that should inform query generation.

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
        response = _call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=300,
            json_mode=True,
        )

        result = json.loads(response)
        return result

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Principle extraction parse error: {e}")
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
        response = _call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,  # Slight creativity for query variation
            max_tokens=400,
            json_mode=True,
        )

        result = json.loads(response)
        return result.get("queries", [])

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Multi-query generation parse error: {e}")
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
        response = _call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,  # Slight creativity for varied sub-questions
            max_tokens=400,
            json_mode=True,
        )

        result = json.loads(response)
        sub_questions = result.get("sub_questions", [])

        if not sub_questions or not isinstance(sub_questions, list):
            logger.warning("[decompose] No sub-questions extracted, using original")
            return [query], response

        logger.info(f"[decompose] Generated {len(sub_questions)} sub-questions")
        return sub_questions, response

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[decompose] Parse error: {e}, using original query")
        return [query], str(e)


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
