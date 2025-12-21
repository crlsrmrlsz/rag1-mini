# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG1-Mini is a Retrieval-Augmented Generation pipeline for creating a hybrid neuroscientist + philosopher AI. It processes 19 books (8 neuroscience, 11 philosophy/wisdom) through a 6-stage pipeline: extraction, cleaning, segmentation, chunking, embedding, and vector storage.

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
python -m src.run_stage_6_weaviate     # Upload to Weaviate (requires running Weaviate)
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

### Commits
- Always commit changes after completing a task or significant modification (do not wait for user to ask)

### Learning Mode
This is a learning project. For every code change:
1. **Explain the theory** - What RAG concept does this implement? (1 paragraph)
2. **Show the library** - What libraries are used and why? (1 paragraph)
3. **Trace the flow** - How does data move through the change? (1 paragraph)

## Key Modules

| Module | Purpose | Interface |
|--------|---------|-----------|
| `src/extractors/docling_parser.py` | PDF extraction | `extract_pdf(path) -> str` |
| `src/processors/text_cleaner.py` | Markdown cleaning | `run_structural_cleaning(text, name) -> (str, log)` |
| `src/processors/nlp_segmenter.py` | Sentence segmentation | `segment_document(text, name) -> List[Dict]` |
| `src/ingest/naive_chunker.py` | Section chunking | `run_section_chunking() -> Dict[str, int]` |
| `src/ingest/embed_texts.py` | Embedding API | `embed_texts(texts) -> List[List[float]]` |
| `src/vector_db/weaviate_client.py` | Weaviate storage | `upload_embeddings(client, name, chunks) -> int` |
| `src/preprocessing/query_classifier.py` | Query preprocessing | `preprocess_query(query) -> PreprocessedQuery` |
| `src/generation/answer_generator.py` | Answer synthesis | `generate_answer(query, chunks) -> GeneratedAnswer` |

## Configuration (src/config.py)

- `MAX_CHUNK_TOKENS = 800` - Target chunk size
- `OVERLAP_SENTENCES = 2` - Sentence overlap between chunks
- `SPACY_MODEL = "en_core_sci_sm"` - NLP model (fallback: en_core_web_sm)
- `TOKENIZER_MODEL = "text-embedding-3-large"` - Token counting
- `WEAVIATE_HOST = "localhost"` - Weaviate server host
- `WEAVIATE_HTTP_PORT = 8080` - REST API port
- `WEAVIATE_GRPC_PORT = 50051` - gRPC port (v4 client)
- `get_collection_name()` - Auto-generates collection name from strategy/model/version
- `PREPROCESSING_MODEL = "openai/gpt-5-nano"` - Query classification model
- `GENERATION_MODEL = "openai/gpt-5-mini"` - Answer generation model

## Memory Bank

The `memory-bank/` directory contains project context:
- `project-status.md` - Pipeline status and overview
- `model-selection.md` - Model research and pricing
- `rag-improve-research.md` - RAG improvement strategies
- `evaluation-history.md` - RAGAS runs with configs and metrics

Update these files when making significant changes to maintain project continuity.

## Current Tasks
<!-- UPDATE THIS SECTION: After completing work, move tasks to "Completed Recently" and add new tasks from plan discussions. Keep only last 5 completed items. -->

### Active
- [ ] Run alpha tuning experiments (0.3, 0.5, 0.7) with RAGAS
- [ ] Query decomposition for MULTI_HOP queries

### Next
- [ ] Contextual retrieval (prepend LLM context to chunks)

### Completed Recently
- Redesigned UI with pipeline-ordered sidebar stages and Pipeline Log tab (Dec 21)
- Added full prompt logging for preprocessing, generation, and reranking (Dec 21)
- Implemented query preprocessing (classification + step-back prompting) (Dec 20)
- Implemented LLM answer generation from retrieved chunks (Dec 20)
- Improved prompts for integrated human behavior understanding (Dec 20)
