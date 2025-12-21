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

from src.config import GENERATION_MODEL
from src.preprocessing import QueryType
from src.utils.file_utils import setup_logging
from src.utils.openrouter_client import call_chat_completion

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
        system_prompt_used: The system prompt sent to LLM (for logging).
        user_prompt_used: The user prompt with context sent to LLM (for logging).
    """

    answer: str
    sources_used: List[int] = field(default_factory=list)
    model: str = ""
    generation_time_ms: float = 0.0
    query_type: Optional[QueryType] = None
    system_prompt_used: Optional[str] = None
    user_prompt_used: Optional[str] = None


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_FACTUAL = """You are a knowledgeable guide to neuroscience and philosophical wisdom.

Answer the user's factual question based ONLY on the provided context.

Rules:
- Provide accurate, clear information with source citations [1], [2], etc.
- If context contains both scientific and philosophical perspectives, include both
- If the context doesn't contain the answer, say so honestly
- Do NOT invent information beyond what's in the context

Context will be provided as numbered passages."""

SYSTEM_PROMPT_OPEN_ENDED = """You are an integrated guide to understanding human nature, drawing on:
- NEUROSCIENCE: How the brain works - mechanisms, circuits, neurotransmitters, evolution
- PHILOSOPHY: How to live well - wisdom traditions, practical ethics, meaning, self-mastery

Your purpose: Help the user understand themselves and human behavior by bridging scientific
knowledge with practical wisdom. Science tells us HOW we work; philosophy guides us in
what to DO with that knowledge.

Approach:
- Draw on whatever is relevant in the context - science, philosophy, or both
- When both apply: show how they illuminate each other
- Be substantive and thoughtful - these questions deserve depth
- Cite sources [1], [2] so the user can explore further
- If context is limited, work with what's there and note gaps honestly

The goal is insight and understanding, not just information.

Context will be provided as numbered passages."""

SYSTEM_PROMPT_MULTI_HOP = """You are an analytical guide to human nature, integrating neuroscience and philosophy.

The user is asking a question that requires synthesis across multiple concepts or perspectives.

Approach:
- Address each aspect of the question systematically
- Show connections: how does understanding X illuminate Y?
- When comparing domains: note where science and philosophy complement, overlap, or diverge
- Be honest about complexity - human nature resists simple answers
- Cite sources [1], [2] for each substantive point

The goal is to help the user see the bigger picture by weaving together different threads of knowledge.

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
    answer_text = call_chat_completion(
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
        system_prompt_used=system_prompt,
        user_prompt_used=user_message,
    )
