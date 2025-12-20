"""Answer generation from retrieved chunks using LLM.

## RAG Theory: Answer Generation

The generation phase synthesizes retrieved context into coherent answers.
This is where RAG differs from pure retrieval - instead of showing raw chunks,
we use an LLM to:

1. **Synthesize** information across multiple chunks
2. **Filter** irrelevant portions of retrieved context
3. **Cite sources** for transparency and verifiability
4. **Adapt tone** based on query type (factual vs philosophical)

## Library Usage

Uses OpenRouter API via `requests` (same pattern as preprocessing module).
Prompt templates are customized per query type for optimal responses.

## Data Flow

1. Retrieved chunks from Weaviate search
2. Format chunks as numbered context for the LLM
3. Apply query-type-specific system prompt
4. LLM generates answer with source citations
5. Return GeneratedAnswer with text and metadata
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import requests

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GENERATION_MODEL,
)
from src.preprocessing import QueryType
from src.utils.file_utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class GeneratedAnswer:
    """Result of answer generation.

    Attributes:
        answer: The generated answer text.
        sources_used: List of chunk indices (1-based) cited in the answer.
        model: Model ID used for generation.
        generation_time_ms: Time taken in milliseconds.
        query_type: The query type used for prompt selection.
    """

    answer: str
    sources_used: List[int] = field(default_factory=list)
    model: str = ""
    generation_time_ms: float = 0.0
    query_type: Optional[QueryType] = None


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_FACTUAL = """You are a knowledgeable assistant with access to neuroscience textbooks and philosophy books.

Answer the user's question based ONLY on the provided context. Be concise and accurate.

Rules:
- If the answer is in the context, provide it clearly with source citations [1], [2], etc.
- If the context doesn't contain the answer, say "I don't have enough information to answer this."
- Do NOT make up information. Only use what's in the context.
- Cite your sources by number [1], [2], etc.

Context will be provided as numbered passages."""

SYSTEM_PROMPT_OPEN_ENDED = """You are a wise advisor with access to philosophical and wisdom traditions (Stoicism, Taoism, Confucianism) as well as neuroscience insights on human behavior.

The user is asking an open-ended or life-wisdom question. Synthesize insights from the provided sources to offer a thoughtful, nuanced response.

Guidelines:
- Draw on multiple sources when relevant, weaving together different perspectives
- Acknowledge the complexity of the question rather than oversimplifying
- Cite specific sources [1], [2] when referencing ideas
- Be warm but not preachy; share wisdom without lecturing
- If sources contain relevant wisdom, share it; if not, acknowledge limitations

Context will be provided as numbered passages from philosophy and neuroscience texts."""

SYSTEM_PROMPT_MULTI_HOP = """You are an analytical assistant with access to neuroscience and philosophy texts.

The user is asking a question that may require synthesizing information from multiple sources or making comparisons.

Guidelines:
- Address each aspect of the question systematically
- Compare and contrast when multiple perspectives are present
- Cite sources [1], [2] for each point made
- Draw explicit connections between concepts from different sources
- If information is incomplete for comparison, acknowledge what's missing

Context will be provided as numbered passages."""


def _get_system_prompt(query_type: QueryType) -> str:
    """Get the appropriate system prompt for the query type."""
    prompts = {
        QueryType.FACTUAL: SYSTEM_PROMPT_FACTUAL,
        QueryType.OPEN_ENDED: SYSTEM_PROMPT_OPEN_ENDED,
        QueryType.MULTI_HOP: SYSTEM_PROMPT_MULTI_HOP,
    }
    return prompts.get(query_type, SYSTEM_PROMPT_FACTUAL)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as numbered context for the LLM.

    Args:
        chunks: List of chunk dictionaries from search results.

    Returns:
        Formatted context string with numbered passages.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        # Extract book and section info
        book_id = chunk.get("book_id", "Unknown")
        section = chunk.get("section", "")
        text = chunk.get("text", "")

        # Format: [1] Book (Section): Text
        header = f"[{i}] {book_id}"
        if section:
            header += f" ({section})"

        context_parts.append(f"{header}:\n{text}")

    return "\n\n---\n\n".join(context_parts)


def _call_chat_completion(
    messages: List[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Call OpenRouter chat completion API.

    Args:
        messages: List of message dicts with role and content.
        model: Model ID (e.g., "openai/gpt-5-mini").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

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

    max_retries = 3
    backoff_base = 1.5

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)

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


def _extract_source_citations(answer: str, num_chunks: int) -> List[int]:
    """Extract source citation numbers from the answer text.

    Args:
        answer: The generated answer containing [1], [2] style citations.
        num_chunks: Total number of chunks provided (for validation).

    Returns:
        List of unique citation numbers found (1-based).
    """
    import re

    citations = set()
    # Match [1], [2], etc.
    for match in re.finditer(r'\[(\d+)\]', answer):
        num = int(match.group(1))
        if 1 <= num <= num_chunks:
            citations.add(num)

    return sorted(citations)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    query_type: QueryType = QueryType.FACTUAL,
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> GeneratedAnswer:
    """Generate an answer from retrieved chunks using an LLM.

    Synthesizes information from retrieved context to answer the user's query.
    Uses query-type-specific prompts for optimal response style.

    Args:
        query: The user's original query.
        chunks: List of chunk dictionaries from search results.
            Each chunk should have 'text', 'book_id', and optionally 'section'.
        query_type: The classified query type (affects prompt selection).
        model: Override model (defaults to GENERATION_MODEL from config).
        temperature: Sampling temperature (higher = more creative).

    Returns:
        GeneratedAnswer with the answer text and metadata.

    Raises:
        requests.RequestException: On API errors after retries.
        ValueError: If no chunks are provided.

    Example:
        >>> chunks = search_chunks("What is serotonin?", top_k=5)
        >>> answer = generate_answer("What is serotonin?", chunks)
        >>> print(answer.answer)
        "Serotonin is a neurotransmitter... [1]"
    """
    if not chunks:
        return GeneratedAnswer(
            answer="No relevant context was found to answer this question.",
            sources_used=[],
            model=model or GENERATION_MODEL,
            query_type=query_type,
        )

    model = model or GENERATION_MODEL
    start_time = time.time()

    # Format context from chunks
    context = _format_context(chunks)

    # Build messages
    system_prompt = _get_system_prompt(query_type)
    user_message = f"""Context:
{context}

Question: {query}

Please answer based on the context above, citing sources by number [1], [2], etc."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Generate answer
    logger.info(f"Generating answer with {model} for {query_type.value} query")
    answer_text = _call_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=1024,
    )

    # Extract citations
    sources_used = _extract_source_citations(answer_text, len(chunks))

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"Answer generated in {elapsed_ms:.0f}ms, cited {len(sources_used)} sources")

    return GeneratedAnswer(
        answer=answer_text,
        sources_used=sources_used,
        model=model,
        generation_time_ms=elapsed_ms,
        query_type=query_type,
    )
