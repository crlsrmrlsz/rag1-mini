# Section Chunking (Baseline)

[← Chunking Overview](README.md) | [Home](../../README.md)

This is the baseline chunking strategy, leveraging the structure authors have already created. It operates on a key assumption: **each section contains a single coherent subject, and each paragraph within that section represents a self-contained idea**. By respecting these natural boundaries, the chunker splits documents into chunks with a maximum 800-token size while maintaining sentence overlap for context continuity.


### Algorithm

```
For each document:
  1. Load NLP-segmented paragraphs (spaCy sentence boundaries)
  2. Initialize: current_chunk = [], current_context = None

  For each paragraph:
    If context changed (new section):
      Save current_chunk
      Start new chunk (no overlap across sections)

    For each sentence:
      If (current_chunk + sentence) <= MAX_TOKENS:
        Append sentence to chunk
      Else:
        Save current_chunk
        Start new chunk with last 2 sentences (overlap)
```

### Core Function

```python
# src/rag_pipeline/chunking/section_chunker.py

def create_chunks_from_paragraphs(
    paragraphs: List[Dict],
    book_name: str,
    max_tokens: int = 800,
    overlap_sentences: int = 2
) -> List[Dict]:
```

Each chunk includes metadata: `chunk_id`, `book_id`, `context` (hierarchical path like "Book > Chapter > Section"), `section`, `text`, `token_count`.

### Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Chunk size** | 800 tokens (max) | Upper limit balancing paragraph unity and retrieval performance |
| **Overlap** | 2 sentences | Handles "As mentioned above..." references with minimal redundancy (~50-100 tokens) |
| **Section boundaries** | Hard break | Prevents mixing unrelated content; preserves author's semantic organization |
| **Tokenizer** | tiktoken (text-embedding-3-large) | Exact token count matching embedding model |

### Chunk Size: 800-Token Limit

Research shows optimal chunk size depends on content type and query complexity. [NVIDIA's chunking benchmark](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/) tested sizes from 128 to 2,048 tokens and found 512-1024 tokens optimal for complex analytical queries, while page-level chunking achieved the highest overall accuracy (0.648). [Academic research on long-document retrieval](https://arxiv.org/html/2505.21700v2) confirms this pattern: smaller chunks (64-128 tokens) work best for factoid queries with concise answers, but larger chunks (512-1024 tokens) significantly improve retrieval for technical content—TechQA accuracy jumped from 4.8% at 64 tokens to 71.5% at 1024 tokens. For content requiring broader contextual understanding like NarrativeQA, performance improved from 4.2% to 10.7% as chunk size increased from 64 to 1024 tokens.

Analysis of this corpus reveals distinct patterns between content types:

| Corpus | Avg Section Tokens | Median |
|--------|-------------------|--------|
| **Neuroscience** | 666 | ~500-700 |
| **Philosophy** | 1,427 | varies widely | 

Neuroscience textbooks have well-structured sections averaging 666 tokens—comfortably below the 800-token limit, meaning most conceptual units remain intact within single chunks. Philosophy texts show much higher variance, ranging from aphoristic works like Tao Te Ching (159 avg) and Art of Living (238 avg) that fit easily in single chunks, to essay collections like Seneca's Letters (2,127 avg) and Schopenhauer (2,300+ avg) that require multiple chunks per section.

The 800-token limit represents a balanced estimate for this mixed corpus. The underlying assumption is that well-written paragraphs typically contain single, self-contained ideas—by preserving paragraph unity, each chunk is more likely to represent one coherent concept useful for answering queries. This limit falls within the 512-1024 range that the NVIDIA and academic studies cited above identify as optimal for technical and analytical content, while preserving most neuroscience textbook sections as complete units. For philosophy essays that exceed this limit, the 2-sentence overlap helps maintain some continuity, though advanced techniques like Contextual Chunking or RAPTOR may provide better results for such content. 

This is an upper limit, not a target—actual chunks are often smaller when sections end naturally. Future work could investigate tuning chunk limits per content type: shorter limits for factoid-heavy reference works, longer for essay-style texts requiring extended context. Semantic chunking also enforces this 800-token maximum to prevent oversized segments regardless of similarity scores.

---

### Limitations

- **No document-level context in embeddings**: "The company" doesn't know which company
- **Struggles with cross-section references**: "As Chapter 3 explained..." loses connection
- **Vocabulary mismatch**: Embedding reflects chunk words, not chunk meaning

These limitations motivate Contextual Chunking (adds LLM context) and RAPTOR (hierarchical summaries).

---

## Navigation

**Next:** [Semantic Chunking](semantic-chunking.md) — Embedding-based topic boundaries

**Related:**
- [Contextual Chunking](contextual-chunking.md) — LLM-generated context prepended
- [RAPTOR](raptor.md) — Hierarchical summarization alternative
- [Chunking Overview](README.md) — Strategy comparison
