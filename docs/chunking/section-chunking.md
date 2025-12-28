# Section Chunking

The baseline chunking strategy. Splits documents into fixed-size chunks while respecting section boundaries and maintaining sentence overlap for context continuity.

## TL;DR

800-token chunks with 2-sentence overlap between consecutive chunks within the same section. No LLM calls — pure algorithmic splitting based on NLP sentence segmentation.

## The Problem

Naive chunking (split every N tokens) creates several issues:

1. **Mid-sentence splits**: "The prefrontal cortex regulates" / "emotional responses through..."
2. **Context loss**: A chunk about "the study" loses which study
3. **Section boundary violations**: Mixing conclusion with next chapter's intro

## The Solution

### Algorithm

```
For each document:
  1. Load NLP-segmented paragraphs (sentences already split)
  2. Initialize: current_chunk = [], current_context = None

  For each paragraph:
    If context changed (new section):
      Save current_chunk
      Start new chunk (no overlap across sections)

    For each sentence:
      If (current_chunk + sentence) ≤ MAX_TOKENS:
        Append sentence to chunk
      Else:
        Save current_chunk
        Start new chunk with last 2 sentences (overlap)
```

### Key Design Decisions

**Why 800 tokens?**
- Matches embedding model sweet spot (text-embedding-3-large)
- Large enough for complete thoughts
- Small enough for precise retrieval
- Leaves room for reranking context windows

**Why 2-sentence overlap?**
- Maintains continuity for split concepts
- Minimal redundancy (~50-100 tokens)
- Handles "As mentioned above..." references

**Why respect section boundaries?**
- Sections are semantic units (author's organization)
- Prevents mixing unrelated content
- Preserves hierarchical context metadata

## Implementation Details

### Core Function

```python
# src/rag_pipeline/chunking/section_chunker.py

def create_chunks_from_paragraphs(
    paragraphs: List[Dict],
    book_name: str,
    max_tokens: int = 800,
    overlap_sentences: int = 2
) -> List[Dict]:
    """
    Process paragraphs sequentially to create chunks with overlap.

    Algorithm:
    1. Read paragraphs in order (preserves reading sequence)
    2. When context changes → save chunk, start new, clear overlap
    3. For each sentence:
       - Add to chunk if it fits
       - If doesn't fit → save chunk, start new with overlap
       - If sentence too large → split it
    4. Overlap: Last N sentences initialize next chunk (same section)
    """
```

### Handling Oversized Sentences

Some sentences (especially in academic text) exceed 800 tokens. The chunker handles this gracefully:

```python
def split_oversized_sentence(sentence: str, max_tokens: int) -> List[str]:
    """
    Split a sentence that exceeds token limit.

    Strategy:
    1. Try splitting by punctuation ("; ", ": ", ", ")
    2. Fallback to word boundary splitting
    """
    # If sentence fits, no splitting needed
    if count_tokens(sentence) <= max_tokens:
        return [sentence]

    # Try splitting by punctuation marks (in priority order)
    for separator in ["; ", ": ", ", "]:
        if separator not in sentence:
            continue
        # ... split and reassemble
```

### Chunk Metadata

Each chunk includes rich metadata for retrieval:

```python
def _create_chunk_dict(text, context, book_name, chunk_id):
    return {
        "chunk_id": f"{book_name}::chunk_{chunk_id}",
        "book_id": book_name,
        "context": context,      # "BookTitle > Chapter > Section"
        "section": parse_section_name(context),  # "Section"
        "text": text,
        "token_count": count_tokens(text),
        "chunking_strategy": "sequential_overlap_2"
    }
```

The `context` field preserves the full hierarchical path, enabling:
- Filtering by book or chapter
- Display of breadcrumb navigation
- Cross-referencing within sections

## When to Use

**Good for:**
- Initial development and iteration
- Cost-sensitive deployments (no LLM calls)
- Well-structured documents with clear sections
- Specific fact retrieval ("What did X say about Y?")

**Limitations:**
- Chunks lack document-level context in embeddings
- "The company" without knowing which company
- Struggles with cross-section references

## Results

See [Evaluation Results](../evaluation/results.md) for RAGAS metrics comparing section chunking against contextual and RAPTOR strategies.

## Related

- [Contextual Chunking](contextual-chunking.md) — Adds LLM context to these chunks
- [RAPTOR](raptor.md) — Hierarchical summarization alternative
- [NLP Segmentation](../content-preparation/pdf-extraction.md) — Prerequisite stage
