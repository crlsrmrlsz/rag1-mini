# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG1-Mini is a Retrieval-Augmented Generation pipeline designed for learning and experimentation. It processes PDF documents through an 8-stage pipeline: extraction, cleaning, segmentation, chunking, embedding, vector storage, query/search, and answer generation.

## Environment Setup

```bash
conda activate rag1-mini
```

## Commands

### Run Pipeline Stages

```bash
python -m src.stages.run_stage_1_extraction   # PDF to Markdown
python -m src.stages.run_stage_2_processing   # Markdown cleaning
python -m src.stages.run_stage_3_segmentation # NLP sentence segmentation
python -m src.stages.run_stage_4_chunking     # Section-aware chunking (800 tokens, 2-sentence overlap)
python -m src.stages.run_stage_5_embedding    # Generate embeddings (requires OpenRouter API key)
python -m src.stages.run_stage_6_weaviate     # Upload to Weaviate (requires running Weaviate)
python -m src.stages.run_stage_7_evaluation   # RAGAS evaluation
```

## Code Standards

### Architecture Principles
- **Function-based design**: Use functions as primary interface
- **Classes only for state**: Only use classes when initialization state is needed (e.g., spaCy model uses lazy singleton)
- **Fail-fast error handling**: Let exceptions propagate; do not catch-and-continue
- **Absolute imports**: Always use `from src.module import ...`

### Logging
- Use `logger` from `src.shared.setup_logging()` for all output
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

## Project Structure

The codebase is organized into two main phases for learners:

```
src/
├── content_preparation/          # Phase 1: Documents -> Text (Stages 1-3)
│   ├── extraction/               # Stage 1: PDF -> Markdown
│   ├── cleaning/                 # Stage 2: Clean Markdown
│   └── segmentation/             # Stage 3: Sentence splits
│
├── rag_pipeline/                 # Phase 2: RAG System (Stages 4-8)
│   ├── chunking/                 # Stage 4: Text -> Chunks
│   ├── embedding/                # Stage 5: Chunks -> Vectors
│   ├── indexing/                 # Stage 6: Vector DB
│   ├── retrieval/                # Stage 7: Query -> Chunks
│   │   ├── preprocessing/        # Query transformation
│   │   ├── reranking.py          # Cross-encoder
│   │   └── rrf.py                # RRF merging for decomposition
│   └── generation/               # Stage 8: Chunks -> Answer
│
├── evaluation/                   # RAGAS framework
├── ui/                           # Streamlit app
├── shared/                       # Common utilities
├── stages/                       # Pipeline stage runners
└── config.py                     # Configuration
```

## Key Modules

| Module | Purpose | Interface |
|--------|---------|-----------|
| `src/content_preparation/extraction/docling_parser.py` | PDF extraction | `extract_pdf(path) -> str` |
| `src/content_preparation/cleaning/text_cleaner.py` | Markdown cleaning | `run_structural_cleaning(text, name) -> (str, log)` |
| `src/content_preparation/segmentation/nlp_segmenter.py` | Sentence segmentation | `segment_document(text, name) -> List[Dict]` |
| `src/rag_pipeline/chunking/section_chunker.py` | Section chunking | `run_section_chunking() -> Dict[str, int]` |
| `src/rag_pipeline/embedding/embedder.py` | Embedding API | `embed_texts(texts) -> List[List[float]]` |
| `src/rag_pipeline/indexing/weaviate_client.py` | Weaviate storage | `upload_embeddings(client, name, chunks) -> int` |
| `src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py` | Query preprocessing | `preprocess_query(query) -> PreprocessedQuery` |
| `src/rag_pipeline/generation/answer_generator.py` | Answer synthesis | `generate_answer(query, chunks) -> GeneratedAnswer` |
| `src/shared/openrouter_client.py` | Unified LLM API | `call_chat_completion()`, `call_structured_completion()` |

## Configuration (src/config.py)

- `MAX_CHUNK_TOKENS = 800` - Target chunk size
- `OVERLAP_SENTENCES = 2` - Sentence overlap between chunks
- `SPACY_MODEL = "en_core_sci_sm"` - NLP model (fallback: en_core_web_sm)
- `TOKENIZER_MODEL = "text-embedding-3-large"` - Token counting
- `WEAVIATE_HOST = "localhost"` - Weaviate server host
- `WEAVIATE_HTTP_PORT = 8080` - REST API port
- `WEAVIATE_GRPC_PORT = 50051` - gRPC port (v4 client)
- `get_collection_name()` - Auto-generates collection name from strategy/model/version
- `PREPROCESSING_MODEL` - Query preprocessing model (configurable)
- `GENERATION_MODEL` - Answer generation model (configurable)

## Memory Bank

The `memory-bank/` directory contains project context:
- `project-status.md` - Pipeline status and overview
- `model-selection.md` - Model research and pricing
- `rag-improve-research.md` - RAG improvement strategies
- `evaluation-history.md` - RAGAS runs with configs and metrics
- `rag-improvement-plan.md` - Detailed implementation plan for RAG improvements

Update these files when making significant changes to maintain project continuity.

## Current Tasks
<!-- UPDATE THIS SECTION: After completing work, move tasks to "Completed Recently" and add new tasks from plan discussions. Keep only last 5 completed items. -->

### RAG Improvement Plan (see memory-bank/rag-improvement-plan.md)

**Phase 0: Evaluation CLI** - COMPLETE
- [x] Add --collection argument to run_stage_7_evaluation.py
- [x] Auto-append results to evaluation-history.md
- [x] Update tracking.json with run config

**Phase 1: Preprocessing Strategy Infrastructure** - COMPLETE
- [x] Create strategies.py module with registry pattern
- [x] Add strategy selector to UI Stage 1
- [x] Add --preprocessing argument to evaluation CLI
- [x] Track strategy in evaluation logs

**Phase 2: Remove Classification + Simplify** - COMPLETE (Dec 22)
- [x] Remove query classification (not in original research papers)
- [x] Each strategy applies directly to any query
- [x] Unified answer generation prompt
- [x] Add LLM call logging

**Phase 3: Multi-Query Strategy** - REMOVED (Dec 23)
- Removed: Decomposition strategy subsumes multi-query's domain-targeting
- RRF merging infrastructure retained for decomposition strategy
- See analysis: /.claude/plans/validated-dancing-thacker.md

**Phase 3b: Step-Back → HyDE** - COMPLETE (Dec 23)
- Replaced step_back with HyDE (Hypothetical Document Embeddings, arXiv:2212.10496)
- HyDE generates hypothetical answers for semantic matching (proper RAG research)
- step_back was a reasoning technique adapted for RAG, not from RAG research
- Available strategies now: `none`, `hyde`, `decomposition`

**Phase 4: Query Decomposition** - COMPLETE
- [x] Add DECOMPOSITION_PROMPT and decompose_query() function
- [x] Implement decomposition_strategy in strategies.py
- [x] Register decomposition in STRATEGIES dict
- [x] Update config, CLI, logger, and UI

**Phase 5: Alpha Tuning**
- [ ] Alpha tuning experiments (0.3, 0.5, 0.7) via CLI

**Phase 6: Contextual Retrieval** - COMPLETE (Dec 22)
- [x] Create contextual_chunker.py (prepend LLM context to chunks)
- [x] Add contextual_strategy to strategies.py
- [x] Run via: `python -m src.stages.run_stage_4_chunking --strategy contextual`

**Phase 7: RAPTOR** (Hierarchical summarization, +20% comprehension)
- [ ] Create raptor_chunker.py (hierarchical tree building)
- [ ] Add raptor_strategy to strategies.py
- [ ] Add RAPTOR query strategy

**Phase 8: GraphRAG** (Neo4j, +70% comprehensiveness)
- [ ] Add Neo4j to docker-compose.yml
- [ ] Create src/graph/ module (extractor, neo4j_client, query)
- [ ] Create graph extraction and upload stages

**Note:** Evaluation runs via CLI (`python -m src.stages.run_stage_7_evaluation`), not in UI.

### Completed Recently
- Replaced step_back with HyDE (Hypothetical Document Embeddings, arXiv:2212.10496) - proper RAG research technique (Dec 23)
- Removed multi_query strategy: decomposition subsumes its domain-targeting (~380 lines removed) (Dec 23)
- Contextual chunking strategy (Anthropic-style, +35% failure reduction) (Dec 22)
- Domain-agnostic refactoring: removed book categories, diversification, generalized prompts (Dec 22)
- Removed query classification + unified answer prompt (~170 lines removed) (Dec 22)
- Pydantic structured outputs for LLM responses with JSON Schema enforcement (Dec 22)
- Major codebase refactoring: Two-phase architecture (content_preparation/, rag_pipeline/) for pedagogical clarity (Dec 21)
- Unified OpenRouter API client (src/shared/openrouter_client.py) replacing 3 duplicate implementations (Dec 21)
