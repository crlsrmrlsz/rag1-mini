# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG1-Mini is a Retrieval-Augmented Generation pipeline for creating a hybrid neuroscientist + philosopher AI. It processes 19 books (8 neuroscience, 11 philosophy/wisdom) through a 5-stage pipeline: extraction, cleaning, segmentation, chunking, and embedding.

## Environment Setup

```bash
conda activate rag1-mini
```

## Commands

### Run Pipeline Stages

```bash
python -m src.run_stage_1_extraction   # PDF to Markdown
python -m src.run_stage_2_processing   # Markdown cleaning
python -m src.run_stage_3_segmentation # NLP sentence segmentation
python -m src.run_stage_4_chunking     # Section-aware chunking (800 tokens, 2-sentence overlap)
python -m src.run_stage_5_embedding    # Generate embeddings (requires OpenRouter API key)
```

## Code Standards

### Architecture Principles
- **Function-based design**: Use functions as primary interface
- **Classes only for state**: Only use classes when initialization state is needed (e.g., spaCy model uses lazy singleton)
- **Fail-fast error handling**: Let exceptions propagate; do not catch-and-continue
- **Absolute imports**: Always use `from src.module import ...`

### Logging
- Use `logger` from `src.utils.setup_logging()` for all output
- No emoji in log messages
- No `print()` statements

### Docstrings
- Google style docstrings for all public functions
- Include Args, Returns, and Raises sections

## Pipeline Architecture

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
```

## Key Modules

| Module | Purpose | Interface |
|--------|---------|-----------|
| `src/extractors/docling_parser.py` | PDF extraction | `extract_pdf(path) -> str` |
| `src/processors/text_cleaner.py` | Markdown cleaning | `run_structural_cleaning(text, name) -> (str, log)` |
| `src/processors/nlp_segmenter.py` | Sentence segmentation | `segment_document(text, name) -> List[Dict]` |
| `src/ingest/naive_chunker.py` | Section chunking | `run_section_chunking() -> Dict[str, int]` |
| `src/ingest/embed_texts.py` | Embedding API | `embed_texts(texts) -> List[List[float]]` |

## Configuration (src/config.py)

- `MAX_CHUNK_TOKENS = 800` - Target chunk size
- `OVERLAP_SENTENCES = 2` - Sentence overlap between chunks
- `SPACY_MODEL = "en_core_sci_sm"` - NLP model (fallback: en_core_web_sm)
- `TOKENIZER_MODEL = "text-embedding-3-large"` - Token counting

## Memory Bank

The `memory-bank/` directory contains Cline project state:
- `projectbrief.md` - Goals and architecture
- `activeContext.md` - Current state
- `progress.md` - Completed stages

Update these files when making significant changes to maintain project continuity.
