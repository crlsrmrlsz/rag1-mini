# Chunking Strategies

Chunking determines how documents are split before embedding and indexing. This is an **index-time decision** — changing chunking strategy requires re-processing the entire corpus.

## Strategy Comparison

| Strategy | Tokens/Chunk | LLM Calls | Best For |
|----------|--------------|-----------|----------|
| [Section](section-chunking.md) | ~800 | 0 | Baseline, fast iteration |
| [Contextual](contextual-chunking.md) | ~900 | 1 per chunk | Better retrieval, production |
| [RAPTOR](raptor.md) | Variable | Many | Multi-hop reasoning, themes |

## Trade-offs

### Section Chunking (Baseline)
- **Pros**: Fast, no API costs, predictable chunk sizes
- **Cons**: Loses document-level context in embeddings
- **Use when**: Iterating quickly, cost-sensitive, simple queries

### Contextual Chunking
- **Pros**: 35% fewer retrieval failures (Anthropic), disambiguates entities
- **Cons**: LLM cost per chunk, longer indexing time
- **Use when**: Production deployments, ambiguous content

### RAPTOR
- **Pros**: Multi-level abstraction, answers theme questions
- **Cons**: Complex, many LLM calls, larger index
- **Use when**: Questions span multiple sections/documents

## Shared Infrastructure

All chunking strategies share:

1. **Token counting**: `tiktoken` with `text-embedding-3-large` tokenizer
2. **Embedding model**: `text-embedding-3-large` (1536 dimensions)
3. **Weaviate storage**: HNSW index + BM25 hybrid search
4. **Chunk metadata**: `book_id`, `section`, `context` (hierarchical path)

## Running Chunking

```bash
# Baseline
python -m src.stages.run_stage_4_chunking --strategy section

# Contextual (requires section chunks first)
python -m src.stages.run_stage_4_chunking --strategy contextual

# RAPTOR
python -m src.stages.run_stage_4_5_raptor
```

Each strategy outputs to `data/processed/05_final_chunks/{strategy}/`.
