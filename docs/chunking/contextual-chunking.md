# Contextual Chunking

> **Source:** [Anthropic Blog: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | September 2024

Prepends LLM-generated context to each chunk, improving embedding quality by disambiguating entities and situating content within the document.

## TL;DR

For each chunk, an LLM generates a 2-3 sentence context snippet based on neighboring chunks. This snippet is prepended to the chunk text before embedding, reducing retrieval failures by 35% (Anthropic's benchmark).

## The Problem

Traditional chunking loses document-level context:

```
Original chunk:
"The company's revenue grew by 3% in Q2, driven primarily by
expansion into Asian markets."

Problem: Which company? What year? What's the overall trend?
```

When embedded, this chunk is similar to any revenue growth discussion, making precise retrieval difficult.

## The Solution

### Contextualized Chunk

```
"[This chunk is from ACME Corp's 2023 annual report, specifically
the Financial Performance section discussing quarterly results.]
The company's revenue grew by 3% in Q2, driven primarily by
expansion into Asian markets."
```

Now the embedding captures:
- Company identity (ACME Corp)
- Time period (2023, Q2)
- Document section (Financial Performance)
- Content type (annual report)

### Algorithm

```
For each chunk in document:
  1. Gather neighboring chunks (2 before + 2 after)
  2. Build context from: book_name, section_path, neighbors
  3. Call LLM: "Generate 2-3 sentences situating this chunk"
  4. Prepend snippet: "[{snippet}] {original_text}"
  5. Re-compute token count
```

## Implementation Details

### Context Gathering

```python
# src/rag_pipeline/chunking/contextual_chunker.py

def gather_document_context(
    chunks: List[Dict],
    current_index: int,
    neighbor_count: int = 2,
    max_context_tokens: int = 2000,
) -> str:
    """Gather text from neighboring chunks as document context.

    Collects chunks before and after the current chunk to provide
    the LLM with surrounding document context.
    """
    start_idx = max(0, current_index - neighbor_count)
    end_idx = min(len(chunks), current_index + neighbor_count + 1)

    context_parts = []
    for i in range(start_idx, end_idx):
        if i != current_index:
            chunk_text = chunks[i].get("text", "")
            section = chunks[i].get("section", "")
            if chunk_text:
                if section:
                    context_parts.append(f"[{section}] {chunk_text}")
                else:
                    context_parts.append(chunk_text)

    return "\n\n".join(context_parts)
```

### Snippet Generation

```python
def generate_contextual_snippet(
    chunk: Dict,
    document_context: str,
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 100,
) -> str:
    """Generate a contextual snippet for a chunk using LLM."""
    prompt = CONTEXTUAL_PROMPT.format(
        document_context=document_context,
        chunk_text=chunk.get("text", ""),
        book_name=chunk.get("book_id", "Unknown"),
        context_path=chunk.get("context", ""),
    )

    snippet = call_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,  # Some creativity but mostly factual
        max_tokens=max_tokens,
    )
    return snippet.strip()
```

### Chunk Assembly

```python
def contextualize_chunk(chunk: Dict, contextual_snippet: str) -> Dict:
    """Create a contextualized version of a chunk."""
    original_text = chunk.get("text", "")

    if contextual_snippet:
        contextualized_text = f"[{contextual_snippet}] {original_text}"
    else:
        contextualized_text = original_text

    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "book_id": chunk.get("book_id", ""),
        "context": chunk.get("context", ""),
        "section": chunk.get("section", ""),
        "text": contextualized_text,
        "token_count": count_tokens(contextualized_text),
        "chunking_strategy": "contextual",
        "original_text": original_text,
        "contextual_snippet": contextual_snippet,
    }
```

### Design Decisions

**Why 2 neighbors each direction?**
- Captures local context without excessive token costs
- Enough to understand flow and references
- Configurable via `CONTEXTUAL_NEIGHBOR_CHUNKS`

**Why temperature 0.3?**
- Low enough for factual accuracy
- High enough to vary phrasing
- Avoids templated outputs

**Why store original_text separately?**
- Debugging: compare contextual vs original retrieval
- Reprocessing: regenerate snippets without re-chunking

## When to Use

**Good for:**
- Production deployments with quality requirements
- Ambiguous content (pronouns, partial references)
- Multi-document corpora (need to distinguish sources)
- Hybrid search (BM25 benefits from added keywords)

**Limitations:**
- LLM cost per chunk (one call per chunk)
- Indexing time increases significantly
- Snippet quality depends on LLM capability

## Cost Analysis

For 19 books with ~5,000 total chunks:
- Model: `gpt-4o-mini` (~$0.15/1M input, ~$0.60/1M output)
- Input: ~2000 tokens/call (context + prompt)
- Output: ~80 tokens/call (snippet)
- **Total cost**: ~$2-3 for full corpus

## Results

Anthropic reports:
- **35% reduction** in retrieval failures (top-20)
- **67% reduction** with BM25 hybrid + reranking
- Largest gains on ambiguous queries

See [Evaluation Results](../evaluation/results.md) for RAGLab-specific metrics.

## Related

- [Section Chunking](section-chunking.md) — Prerequisite (contextual builds on section chunks)
- [RAPTOR](raptor.md) — Alternative approach using hierarchical summaries
