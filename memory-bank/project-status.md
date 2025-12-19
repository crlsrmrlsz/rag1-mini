# RAG1-Mini Project Status

**Last Updated:** December 19, 2025

## Overview

RAG1-Mini is a Retrieval-Augmented Generation pipeline for creating a **hybrid neuroscientist + philosopher** AI. It processes 19 books (8 neuroscience, 11 philosophy/wisdom) through a 7-stage pipeline to answer questions about human behavior with evidence-based, cross-domain insights.

**Core Goal:** Master RAG pipeline components while building specialized AI that provides grounded, thoughtful answers about human cognition and behavior.

## Current Status: All Stages Complete

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
```

## Content

**Neuroscience (8 books):** Sapolsky, Pinel & Barnes, Eagleman & Downar, Gazzaniga, Tommasi et al., Sapolsky (Determined), Gage & Bernard, Fountoulakis & Nimatoudis

**Wisdom (11 books):** Kahneman, Schopenhauer (multiple), Lao Tzu, Seneca, Confucius, Epictetus (multiple), Marcus Aurelius, Baltasar Gracian

## RAGAS Evaluation Results

- **Faithfulness:** 1.0 (answers grounded in context)
- **Answer Relevancy:** 0.96-1.0 (answers address questions)
- **Test Set:** 10 curated questions (single concept, cross-domain, open-ended)

## Next Steps: Stage 8 Advanced RAG

| Improvement | What | Why |
|-------------|------|-----|
| Contextual Retrieval | Prepend LLM context to chunks | 67% reduction in retrieval failure |
| Semantic Chunking | Chunk by semantic similarity | Preserve complete concepts |
| Increased Top-K + Reranking | Retrieve 20, LLM reranks to 5 | Reduce noise |
| Hybrid Search Tuning | Test alpha values (0.3, 0.5, 0.7) | Balance vector/keyword |

## Run Commands

```bash
conda activate rag1-mini

# Pipeline stages
python -m src.run_stage_1_extraction
python -m src.run_stage_2_processing
python -m src.run_stage_3_segmentation
python -m src.run_stage_4_chunking
python -m src.run_stage_5_embedding
python -m src.run_stage_6_weaviate

# UI
streamlit run src/ui/app.py

# Evaluation (see model-selection.md for model options)
python -m src.run_stage_7_evaluation
```

## Code Standards

See `CLAUDE.md` for complete standards:
- Function-based design (classes only for state)
- Absolute imports (`from src.module import ...`)
- Fail-fast error handling
- Logger only (no print, no emoji)
- Google-style docstrings
