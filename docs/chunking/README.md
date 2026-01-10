# Chunking Strategies

[← Content Preparation](../content-preparation/README.md) | [Home](../../README.md)

Chunking determines how documents are split before embedding and indexing. This is an **index-time decision** — changing chunking strategy requires re-processing the entire corpus.

---

## Why Chunking Matters for RAG

Standard RAG systems treat chunks as independent units, but conceptual content — particularly philosophical arguments and neuroscience explanations — develops across many paragraphs. When a philosopher builds an argument about consciousness over 40 pages, or a neuroscientist traces the evolutionary origins of addiction through multiple chapters, naive chunking destroys the conceptual unity that makes these texts meaningful.

The fundamental insight driving recent RAG improvements is that **retrieval must operate at multiple levels of abstraction simultaneously**. Fine-grained chunks enable precise matching, but hierarchical summaries capture thematic coherence, and contextual enrichment helps embeddings understand what chunks are *about* rather than just what words they contain.

This project implements four chunking strategies that address different aspects of this challenge:

| Strategy | Core Innovation | Research Basis |
|----------|-----------------|----------------|
| **Section** | Respects document structure | Standard RAG baseline |
| **Semantic** | Embedding-based topic boundaries | Chroma research (arXiv:2410.13070) |
| **Contextual** | LLM-generated context prepended | Anthropic (Sep 2024) |
| **RAPTOR** | Hierarchical summary tree | Stanford/Google (ICLR 2024) |

---

## Why Custom Implementation

This project implements custom chunking rather than using LangChain or LlamaIndex. After evaluating both frameworks, custom implementation proved superior for this use case:

### Feature Comparison

| Feature | RAGLab | LangChain | LlamaIndex |
|---------|--------|-----------|------------|
| **Token counting** | tiktoken (exact) | Character-based | Approximate |
| **Section boundaries** | Respects markdown hierarchy | Not built-in | Basic |
| **Sentence overlap** | Semantic units (2 sentences) | Character-based (50 chars) | Character-based |
| **Semantic chunking** | Absolute threshold (0.4) | Percentile-based (experimental) | Percentile-based |
| **Contextual Retrieval** | Anthropic's technique | Not available | Not available |
| **RAPTOR hierarchical** | Full ICLR 2024 implementation | Not available | Not available |
| **Oversized handling** | 3-tier graceful degradation | Basic recursive | Basic |

### Key Decisions

1. **Token-exact sizing**: LangChain's `RecursiveCharacterTextSplitter` uses character count by default. A "100 character" chunk could be 20 tokens or 40 tokens depending on content. RAGLab uses `tiktoken` with the exact embedding model tokenizer (`text-embedding-3-large`), ensuring chunks fit within embedding model sweet spots.

2. **Section-aware boundaries**: Unlike pure sliding window approaches, RAGLab never crosses section boundaries. The author's organization provides natural semantic units that remain robust across query types.

3. **Research implementations**: RAPTOR (ICLR 2024) and Contextual Retrieval (Anthropic, Sep 2024) are cutting-edge techniques not available in standard libraries. Implementing from research papers provides both learning value and production capability.

4. **Absolute thresholds**: LangChain's `SemanticChunker` uses percentile-based breakpoints (95th percentile of cosine distances). RAGLab uses an absolute threshold (0.4) based on Chroma research, providing consistent behavior across documents with varying similarity distributions.

---

## Shared Infrastructure

All chunking strategies share common components:

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **Token counting** | `tiktoken` with `text-embedding-3-large` | Exact token counts matching embedding model |
| **Embedding model** | `text-embedding-3-large` (3072 dims) | State-of-the-art dense retrieval |
| **Vector storage** | Weaviate HNSW index + BM25 | Hybrid search (dense + keyword) |
| **Chunk metadata** | `book_id`, `section`, `context` | Hierarchical path for filtering and display |

### Chunk Schema

Every chunk includes standardized metadata:

```json
{
  "chunk_id": "BookName::chunk_42",
  "book_id": "BookName",
  "context": "BookName > Chapter 3 > Section 2",
  "section": "Section 2",
  "text": "The actual chunk content...",
  "token_count": 750,
  "chunking_strategy": "section"
}
```

The `context` field preserves hierarchical position (Book > Chapter > Section), enabling:
- Scoped retrieval ("only search Chapter 5")
- Answer attribution ("This is from Chapter 3, Section 2")
- Cross-reference tracking

---

## Strategy Comparison

From comprehensive evaluation across 102 configurations (5 chunking strategies × search types × preprocessing strategies × alpha values × top_k):

### Answer Quality Metrics

| Strategy | Answer Correctness | Faithfulness | Relevancy |
|----------|-------------------|--------------|-----------|
| **Contextual** | **59.1%** (1st) | 93.9% | 85.5% |
| RAPTOR | 57.9% (2nd) | **95.2%** (1st) | 81.5% |
| Section | 57.6% (3rd) | 95.0% | **89.1%** (1st) |
| Semantic 0.3 | 54.1% (4th) | 90.2% | 83.0% |
| Semantic 0.75 | 50.8% (5th) | 85.4% | 83.0% |

### Retrieval Metrics

| Strategy | Context Precision | Context Recall | Combined |
|----------|-------------------|----------------|----------|
| Semantic 0.3 | **73.4%** | 93.3% | 83.4% |
| **Contextual** | 71.7% | **96.3%** | **84.0%** |
| Semantic 0.75 | 71.2% | 86.1% | 78.7% |
| Section | 69.1% | 92.9% | 81.0% |
| RAPTOR | 68.1% | 96.1% | 82.1% |

### Key Findings

1. **Recall matters more than precision**: Contextual chunking wins on answer correctness despite lower precision than Semantic 0.3. The generator LLM can filter irrelevant context (low precision is recoverable) but cannot invent missing information (low recall is unrecoverable).

2. **Section provides best consistency**: Smallest performance degradation (-16.6% recall) when moving from simple to complex queries. The author's section organization provides natural semantic boundaries.

3. **RAPTOR excels at faithfulness**: Highest faithfulness (95.2%) because pre-computed summaries are verified abstractions, reducing hallucination risk.

4. **Avoid loose semantic thresholds**: Semantic 0.75 underperforms across all metrics. The loose threshold creates overly broad chunks that miss semantic coherence.

---

## Data Flow

```
Stage 3 Output                    Stage 4: Chunking                    Stage 5+
─────────────────────────────────────────────────────────────────────────────────

data/processed/04_nlp_chunks/     data/processed/05_final_chunks/     → Embedding
    ├── book1.json                    ├── section/                     → Weaviate
    ├── book2.json                    │   ├── book1.json               → Retrieval
    └── ...                           │   └── book2.json
                                      ├── contextual/
         NLP-segmented                │   └── ...
         paragraphs with              ├── semantic_0.4/
         sentence boundaries          │   └── ...
                                      └── raptor/
                                          └── ...

                                      Strategy-specific
                                      chunk outputs
```

### Strategy Dependencies

```
                    ┌─────────────────────────────────────┐
                    │         NLP Chunks (Stage 3)        │
                    │    Paragraphs with sentence lists   │
                    └─────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
     │     Section     │    │    Semantic     │    │     RAPTOR      │
     │   (Baseline)    │    │  (Similarity)   │    │  (Hierarchical) │
     └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                                             │
              │                                    Uses section as leaves
              ▼                                             │
     ┌─────────────────┐                                    │
     │   Contextual    │◄───────────────────────────────────┘
     │  (LLM context)  │
     └─────────────────┘
              │
    Builds on section chunks
```

---

## Running Chunking

```bash
# Section (baseline) - No dependencies
python -m src.stages.run_stage_4_chunking --strategy section

# Semantic - No dependencies, specify threshold
python -m src.stages.run_stage_4_chunking --strategy semantic --threshold 0.4

# Contextual - Requires section chunks first
python -m src.stages.run_stage_4_chunking --strategy contextual

# RAPTOR - Separate pipeline (Stage 4.5)
python -m src.stages.run_stage_4_5_raptor
```

### Output Locations

| Strategy | Output Directory |
|----------|------------------|
| Section | `data/processed/05_final_chunks/section/` |
| Semantic | `data/processed/05_final_chunks/semantic_0.4/` |
| Contextual | `data/processed/05_final_chunks/contextual/` |
| RAPTOR | `data/processed/05_final_chunks/raptor/` |

---

## Strategy Selection Guide

```
Start here:
    │
    ├── Need fast iteration? ──────────────► Section
    │   (No LLM calls, instant processing)
    │
    ├── Production deployment?
    │       │
    │       ├── General Q&A ───────────────► Contextual
    │       │   (Best answer correctness)
    │       │
    │       └── Faithfulness critical? ────► RAPTOR
    │           (Highest grounding, lowest hallucination)
    │
    ├── Single-domain corpus? ─────────────► Semantic 0.3
    │   (Best precision when cross-domain not needed)
    │
    └── Theme/synthesis questions? ────────► RAPTOR + GraphRAG
        (Multi-level abstraction + entity relationships)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/rag_pipeline/chunking/section_chunker.py` | Section-based chunking with overlap |
| `src/rag_pipeline/chunking/semantic_chunker.py` | Embedding similarity boundaries |
| `src/rag_pipeline/chunking/contextual_chunker.py` | LLM context generation |
| `src/rag_pipeline/chunking/raptor/tree_builder.py` | RAPTOR tree construction |
| `src/rag_pipeline/chunking/strategies.py` | Strategy registry and CLI routing |
| `src/config.py` | Chunking parameters (MAX_CHUNK_TOKENS, thresholds) |

---

## Navigation

### Strategy Documentation

- **[Section Chunking](section-chunking.md)** — The baseline: fixed-size with sentence overlap
- **[Semantic Chunking](semantic-chunking.md)** — Embedding-based topic boundaries
- **[Contextual Chunking](contextual-chunking.md)** — LLM-generated context prepended
- **[RAPTOR](raptor.md)** — Hierarchical summarization tree

### Related

- [Preprocessing Strategies](../preprocessing/README.md) — Query-time transformations
- [Evaluation Framework](../evaluation/README.md) — How strategies are compared
