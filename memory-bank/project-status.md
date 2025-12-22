# RAG1-Mini Project Status

**Last Updated:** December 21, 2025

## Overview

RAG1-Mini is a Retrieval-Augmented Generation pipeline for creating a **hybrid neuroscientist + philosopher** AI. It processes 19 books (8 neuroscience, 11 philosophy/wisdom) through an 8-stage pipeline to answer questions about human behavior with evidence-based, cross-domain insights.

**Core Goal:** Master RAG pipeline components while building specialized AI that provides grounded, thoughtful answers about human cognition and behavior.

## Current Status: Phase 1 Infrastructure Complete

| Stage | Description | Output |
|-------|-------------|--------|
| 1. Extraction | PDF to Markdown | 19 MD files |
| 2. Cleaning | Manual review + cleaning | 19 cleaned MD files |
| 3. Segmentation | NLP sentence segmentation | 19 JSON files |
| 4. Chunking | Section-aware (800 tokens, 2-sentence overlap) | 6,245 chunks |
| 5. Embedding | text-embedding-3-large via OpenRouter | 19 JSON files |
| 6. Weaviate | Vector storage (HNSW + cosine) | 6,249 objects |
| 7A. Query | `query_similar()`, `query_hybrid()` | weaviate_query.py |
| 7B. UI | Streamlit interface | src/ui/app.py |
| 7C. RAGAS | Evaluation framework | src/evaluation/ |
| 8A. Preprocessing | Query classification + step-back | src/preprocessing/ |
| 8B. Generation | LLM answer synthesis | src/generation/ |

## Data Flow

```
data/raw/ (19 PDFs)
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
data/processed/05_final_chunks/section/ (6,245 chunks)
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
User Query -> classify_query() -> step_back_prompt() -> search -> generate_answer() -> Answer
```

## Content

**Neuroscience (8 books):** Sapolsky, Pinel & Barnes, Eagleman & Downar, Gazzaniga, Tommasi et al., Sapolsky (Determined), Gage & Bernard, Fountoulakis & Nimatoudis

**Wisdom (11 books):** Kahneman, Schopenhauer (multiple), Lao Tzu, Seneca, Confucius, Epictetus (multiple), Marcus Aurelius, Baltasar Gracian

## RAGAS Evaluation Results (Run 4 - Best)

- **Faithfulness:** 0.927 (answers grounded in context)
- **Answer Relevancy:** 0.787 (answers address questions)
- **Failures:** 1/23 questions (4%)
- **Test Set:** 23 curated questions (neuroscience, philosophy, cross-domain)

**Configuration:** Hybrid search, alpha=0.5, top-k=10, cross-encoder reranking

**Note:** Cross-encoder reranking improves quality but is slow on CPU (~2 min/query). Disabled by default; code preserved for future GPU/API use.

## Stage 8: Query Preprocessing + Answer Generation (Completed Dec 22)

| Component | Purpose | Status |
|-----------|---------|--------|
| Step-Back Prompting | Transform to broader concepts for better retrieval | Complete |
| Multi-Query Generation | Generate 4 targeted queries + RRF merge | Complete |
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
| 3 | Multi-Query Strategy (+RRF merging) | COMPLETE |
| 4 | Query Decomposition (always-on) | COMPLETE |
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
│   └──────────────────┘        │   "step_back": step_back_...,│ │
│            │                  │   "multi_query": multi_...,  │ │
│            ▼                  │   "decomposition": decomp_...│ │
│   ┌──────────────────┐        │ }                            │ │
│   │ UI Dropdown      │        │ def get_strategy(id) -> fn   │ │
│   │ CLI --arg        │        └──────────────────────────────┘ │
│   └────────┬─────────┘                     │                   │
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
- `src/rag_pipeline/retrieval/rrf.py` (RRF merging for multi_query)

**Available strategies (each applies directly to any query):**
- `none` - No transformation, use original query (0 LLM calls)
- `step_back` - Transform to broader concepts for better retrieval (1 LLM call)
- `multi_query` - Generate 4 targeted queries + RRF merge (2 LLM calls)
- `decomposition` - Break into 2-4 sub-questions + RRF merge (1 LLM call)

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
- `src/rag_pipeline/retrieval/preprocessing/schemas.py` - Response models (ClassificationResult, PrincipleExtraction, etc.)

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
