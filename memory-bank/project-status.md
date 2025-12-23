# RAG1-Mini Project Status

**Last Updated:** December 22, 2025

## Overview

RAG1-Mini is a Retrieval-Augmented Generation pipeline designed for learning and experimentation. It processes PDF documents through an 8-stage pipeline to build a searchable knowledge base with AI-powered answers.

**Core Goal:** Master RAG pipeline components while building a practical system for document-based question answering.

## Current Status: Phase 1 Infrastructure Complete

| Stage | Description | Output |
|-------|-------------|--------|
| 1. Extraction | PDF to Markdown | Markdown files |
| 2. Cleaning | Manual review + cleaning | Cleaned MD files |
| 3. Segmentation | NLP sentence segmentation | JSON files |
| 4. Chunking | Section-aware (800 tokens, 2-sentence overlap) | Semantic chunks |
| 5. Embedding | text-embedding-3-large via OpenRouter | Embedding files |
| 6. Weaviate | Vector storage (HNSW + cosine) | Vector objects |
| 7A. Query | `query_similar()`, `query_hybrid()` | weaviate_query.py |
| 7B. UI | Streamlit interface | src/ui/app.py |
| 7C. RAGAS | Evaluation framework | src/evaluation/ |
| 8A. Preprocessing | Strategy-based query transformation | src/rag_pipeline/retrieval/preprocessing/ |
| 8B. Generation | LLM answer synthesis | src/generation/ |

## Data Flow

```
data/raw/ (PDF documents)
    |
    v  Stage 1: extract_pdf()
data/processed/01_raw_extraction/
    |
    v  Manual review
data/processed/02_manual_review/
    |
    v  Stage 2: run_structural_cleaning()
data/processed/03_markdown_cleaning/
    |
    v  Stage 3: segment_document()
data/processed/04_nlp_chunks/
    |
    v  Stage 4: run_section_chunking()
data/processed/05_final_chunks/section/
    |
    v  Stage 5: embed_texts()
data/processed/06_embeddings/
    |
    v  Stage 6: upload_embeddings()
Weaviate: RAG_section800_embed3large_v1
    |
    v  Stage 7: Query + UI + Evaluation
    |
    v  Stage 8: Preprocessing + Generation
User Query -> preprocess_query(strategy) -> search -> generate_answer() -> Answer
```

## RAGAS Evaluation

The pipeline includes RAGAS-based evaluation metrics:
- **Faithfulness:** Are answers grounded in the retrieved context?
- **Answer Relevancy:** Do answers address the questions?
- **Context Precision:** Are retrieved chunks relevant?

**Configuration:** Hybrid search, alpha=0.5, top-k=10, cross-encoder reranking

**Note:** Cross-encoder reranking improves quality but is slow on CPU (~2 min/query). Disabled by default; code preserved for future GPU/API use.

## Stage 8: Query Preprocessing + Answer Generation (Completed Dec 22)

| Component | Purpose | Status |
|-----------|---------|--------|
| Step-Back Prompting | Transform to broader concepts for better retrieval | Complete |
| Query Decomposition | Break into sub-questions + RRF merge | Complete |
| Answer Generator | Synthesize LLM answer from retrieved chunks | Complete |
| LLM Call Logging | Log all LLM calls with model and char counts | Complete |

**Key Modules:**
- `src/rag_pipeline/retrieval/preprocessing/` - Strategy-based query transformation
- `src/rag_pipeline/generation/` - LLM answer synthesis with source citations

**Design Decisions (Dec 22):**
- Removed query classification (not in original research papers)
- Each strategy applies its transformation directly to any query
- Unified answer generation prompt (works for all query types)
- LLM call logging: `[LLM] model=X chars_in=Y chars_out=Z`

## RAG Improvement Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Evaluation CLI (--collection, auto-logging) | COMPLETE |
| 1 | Preprocessing Strategy Infrastructure | COMPLETE |
| 2 | Remove Classification + Simplify (Dec 22) | COMPLETE |
| 3 | Multi-Query Strategy | REMOVED (Dec 23) - Subsumed by decomposition |
| 4 | Query Decomposition (always-on) | COMPLETE |
| 2.5 | Domain-Agnostic Refactoring (Dec 22) | COMPLETE |
| 5 | Quick Wins (lost-in-middle, alpha tuning) | TODO |
| 6 | Contextual Chunking (+35% failure reduction) | TODO |
| 7 | RAPTOR (hierarchical summarization) | TODO |
| 8 | GraphRAG (Neo4j integration) | TODO |

See `memory-bank/rag-improvement-plan.md` for detailed implementation plans.

## Strategy Pattern Architecture

The project uses a **Strategy Pattern with Registry** for modular, testable RAG components. This pattern is implemented for preprocessing and will be applied to chunking, embedding, and retrieval.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY PATTERN FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   config.py                    strategies.py                    │
│   ┌──────────────────┐        ┌──────────────────────────────┐ │
│   │ AVAILABLE_*      │───────▶│ STRATEGIES = {               │ │
│   │ DEFAULT_*        │        │   "none": none_strategy,     │ │
│   └──────────────────┘        │   "hyde": hyde_strategy,     │ │
│            │                  │   "decomposition": decomp_...│ │
│            ▼                  │ }                            │ │
│   ┌──────────────────┐        │ def get_strategy(id) -> fn   │ │
│   │ UI Dropdown      │        └──────────────────────────────┘ │
│   │ CLI --arg        │                       │                   │
│   └────────┬─────────┘                       │                   │
│            │                               │                   │
│            └──────────────────▶ dispatcher() ◀─────────────────┘
│                                     │                          │
│                                     ▼                          │
│                            Result dataclass                    │
│                            (with strategy_used)                │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Config** | `src/config.py` | `AVAILABLE_*_STRATEGIES` list, `DEFAULT_*_STRATEGY` |
| **Registry** | `src/*/strategies.py` | `STRATEGIES` dict mapping ID → function |
| **Dispatcher** | Main module | `process_*(strategy=...)` routes to correct function |
| **Result** | Dataclass | Contains `strategy_used` field for tracking |
| **UI** | `src/ui/app.py` | Dropdown populated from `AVAILABLE_*` |
| **CLI** | `src/run_stage_*.py` | `--strategy` argument with choices |
| **Logging** | `src/utils/query_logger.py` | Records `strategy` in JSON logs |

### Implemented: Preprocessing Strategies

**Files:**
- `src/config.py:AVAILABLE_PREPROCESSING_STRATEGIES`
- `src/rag_pipeline/retrieval/preprocessing/strategies.py` (registry)
- `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py` (dispatcher)
- `src/rag_pipeline/retrieval/rrf.py` (RRF merging for decomposition)

**Available strategies (each applies directly to any query):**
- `none` - No transformation, use original query (0 LLM calls)
- `hyde` - Generate hypothetical answer for semantic matching (1 LLM call, 1 search) [arXiv:2212.10496]
- `decomposition` - Break into 2-4 sub-questions + RRF merge (1 LLM call, 3-4 searches) [arXiv:2507.00355]

### To Implement: Chunking Strategies

Same pattern for `src/ingest/chunking_strategies.py`:
- `naive` - Current 800-token section-aware chunks
- `contextual` - Anthropic-style context prepending
- `raptor` - Hierarchical summarization tree

### To Implement: Embedding Strategies

Same pattern for `src/ingest/embedding_strategies.py`:
- `text-embedding-3-large` - Current OpenAI model
- `voyage-3` - Higher quality, different pricing

### To Implement: Retrieval Strategies

Same pattern for `src/vector_db/retrieval_strategies.py`:
- `vector` - Pure semantic search
- `hybrid` - BM25 + vector with alpha
- `graphrag` - Graph-augmented retrieval

## LLM Response Validation

Uses Pydantic schemas for type-safe LLM outputs with JSON Schema enforcement:
- `src/shared/schemas.py` - `get_openrouter_schema()` utility
- `src/rag_pipeline/retrieval/preprocessing/schemas.py` - Response models (DecompositionResult)

Key function: `call_structured_completion(messages, model, response_model)` in `openrouter_client.py`

## Run Commands

```bash
conda activate rag1-mini

# Pipeline stages
python -m src.stages.run_stage_1_extraction
python -m src.stages.run_stage_2_processing
python -m src.stages.run_stage_3_segmentation
python -m src.stages.run_stage_4_chunking
python -m src.stages.run_stage_5_embedding
python -m src.stages.run_stage_6_weaviate

# UI
streamlit run src/ui/app.py

# Evaluation (see model-selection.md for model options)
python -m src.stages.run_stage_7_evaluation
```

## Code Standards

See `CLAUDE.md` for complete standards:
- Function-based design (classes only for state)
- Absolute imports (`from src.module import ...`)
- Fail-fast error handling
- Logger only (no print, no emoji)
- Google-style docstrings
