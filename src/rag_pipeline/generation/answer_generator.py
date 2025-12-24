"""Answer generation from retrieved chunks using LLM.

## RAG Theory: Answer Generation

The generation phase synthesizes retrieved context into coherent answers.
This is where RAG differs from pure retrieval - instead of showing raw chunks,
we use an LLM to:

1. **Synthesize** information across multiple chunks
2. **Filter** irrelevant portions of retrieved context
3. **Cite sources** for transparency and verifiability
4. **Integrate perspectives** from diverse sources

## Library Usage

Uses OpenRouter API via `requests` (same pattern as preprocessing module).
A single unified prompt handles all query types effectively.

## Data Flow

1. Retrieved chunks from Weaviate search
2. Format chunks as numbered context for the LLM
3. Apply unified system prompt
4. LLM generates answer with source citations
5. Return GeneratedAnswer with text and metadata
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.config import GENERATION_MODEL
from src.shared.files import setup_logging
from src.shared.openrouter_client import call_chat_completion

logger = setup_logging(__name__)


@dataclass
class GeneratedAnswer:
    """Result of answer generation.

    Attributes:
        answer: The generated answer text.
        sources_used: List of chunk indices (1-based) cited in the answer.
        model: Model ID used for generation.
        generation_time_ms: Time taken in milliseconds.
        system_prompt_used: The system prompt sent to LLM (for logging).
        user_prompt_used: The user prompt with context sent to LLM (for logging).
    """

    answer: str
    sources_used: List[int] = field(default_factory=list)
    model: str = ""
    generation_time_ms: float = 0.0
    system_prompt_used: Optional[str] = None
    user_prompt_used: Optional[str] = None


# =============================================================================
# DOMAIN CLASSIFICATION
# =============================================================================

# Keywords to identify neuroscience sources (default: philosophy/wisdom)
NEUROSCIENCE_KEYWORDS = [
    "behave", "biopsychology", "brain", "cognitive", "neuroscience",
    "psychobiology", "determined", "sapolsky", "eagleman", "gazzaniga",
    "kahneman", "thinking fast", "pinel", "barnes", "downar", "tommasi",
    "nadel", "fountoulakis", "gage", "bernard",
]


def _get_domain(book_id: str) -> str:
    """Classify a book as neuroscience or philosophy/wisdom.

    Args:
        book_id: The book identifier from chunk metadata.

    Returns:
        'neuroscience' or 'philosophy' based on keyword matching.
    """
    book_lower = book_id.lower()
    for keyword in NEUROSCIENCE_KEYWORDS:
        if keyword in book_lower:
            return "neuroscience"
    return "philosophy"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a knowledgeable assistant that synthesizes cross-domain knowledge,
bridging neuroscience and philosophical wisdom.

The context includes sources from two domains:
- [NEURO] = Neuroscience sources (brain mechanisms, biology, cognitive science)
- [WISDOM] = Philosophy/wisdom sources (Stoics, Schopenhauer, Confucius, Lao Tzu, etc.)

RESPONSE STRUCTURE:
Organize your answer in three sections:

## 🧠 Scientific Understanding
Explain what neuroscience and cognitive science reveal about this topic.
Cite [NEURO] sources. Focus on mechanisms, brain regions, biological processes.

## 📜 Practical Wisdom
Explain what philosophical traditions teach about this topic.
Cite [WISDOM] sources. Focus on principles, practices, and actionable advice.

## 🔗 Convergence & Divergence
Analyze how these perspectives relate:
- Where do ancient wisdom and modern neuroscience **converge**?
- Where do they **diverge** or offer different insights?
- Does the philosophical advice have a grounded neuroscientific basis?

GUIDELINES:
- Each section should be substantive - questions deserve depth
- If only one domain has relevant sources, write that section and note the gap
- The synthesis section should be analytical, not just summary
- Cite sources [1], [2] so users can explore further"""


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as numbered context for the LLM.

    Each chunk is labeled with its domain ([NEURO] or [WISDOM]) to help
    the LLM organize its response by perspective.

    Args:
        chunks: List of chunk dictionaries from search results.

    Returns:
        Formatted context string with numbered, domain-labeled passages.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        # Extract book and section info
        book_id = chunk.get("book_id", "Unknown")
        section = chunk.get("section", "")
        text = chunk.get("text", "")

        # Add domain label for structured response
        domain = _get_domain(book_id)
        domain_label = "[NEURO]" if domain == "neuroscience" else "[WISDOM]"

        # Format: [1] [NEURO] Book (Section): Text
        header = f"[{i}] {domain_label} {book_id}"
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
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> GeneratedAnswer:
    """Generate an answer from retrieved chunks using an LLM.

    Synthesizes information from retrieved context to answer the user's query.
    Uses a unified prompt that handles all query types effectively.

    Args:
        query: The user's original query.
        chunks: List of chunk dictionaries from search results.
            Each chunk should have 'text', 'book_id', and optionally 'section'.
        model: Override model (defaults to GENERATION_MODEL from config).
        temperature: Sampling temperature (higher = more creative).

    Returns:
        GeneratedAnswer with the answer text and metadata.

    Raises:
        requests.RequestException: On API errors after retries.

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
        )

    model = model or GENERATION_MODEL
    start_time = time.time()

    # Format context from chunks
    context = _format_context(chunks)

    # Build messages
    user_message = f"""Context:
{context}

Question: {query}

Please answer based on the context above, citing sources by number [1], [2], etc."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Generate answer
    logger.info(f"Generating answer with {model}")
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
        system_prompt_used=SYSTEM_PROMPT,
        user_prompt_used=user_message,
    )
