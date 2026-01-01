# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGLab is a Retrieval-Augmented Generation pipeline designed for learning and experimentation. It processes PDF documents through an 8-stage pipeline: extraction, cleaning, segmentation, chunking, embedding, vector storage, query/search, and answer generation.

## Environment Setup

```bash
conda activate raglab
```

## Commands

### Run Pipeline Stages

```bash
python -m src.stages.run_stage_1_extraction     # PDF to Markdown
python -m src.stages.run_stage_2_processing     # Markdown cleaning
python -m src.stages.run_stage_3_segmentation   # NLP sentence segmentation
python -m src.stages.run_stage_4_chunking       # Section-aware chunking (800 tokens, 2-sentence overlap)
python -m src.stages.run_stage_5_embedding      # Generate embeddings (requires OpenRouter API key)
python -m src.stages.run_stage_6_weaviate       # Upload to Weaviate (requires running Weaviate)
python -m src.stages.run_stage_7_evaluation     # RAGAS evaluation
```

### GraphRAG Pipeline (Knowledge Graph + Communities)

GraphRAG creates a knowledge graph from your corpus and uses Leiden community detection for global queries.
**Important:** There are two extraction paths - choose ONE:

```bash
# === OPTION A: Auto-Tuning (Recommended) ===
# Discovers entity types FROM your corpus content
python -m src.stages.run_stage_4_5_autotune --strategy section

# === OPTION B: Predefined Types ===
# Uses hardcoded entity types from src/config.py
python -m src.stages.run_stage_4_6_graph_extract --strategy section

# === Re-Consolidate with Stratified Algorithm (for mixed corpora) ===
# Balances entity types across neuroscience vs philosophy books
python -m src.stages.run_stage_4_5_autotune --reconsolidate stratified

# === Then upload to Neo4j + run Leiden (same for both options) ===
docker compose up -d neo4j   # Start Neo4j if not running
python -m src.stages.run_stage_6b_neo4j

# === Query with graphrag strategy ===
# UI: Select "graphrag" in preprocessing dropdown
# CLI (with hybrid search - recommended for graphrag):
python -m src.stages.run_stage_7_evaluation --search-type hybrid --preprocessing graphrag
```

**Consolidation Strategies:**
- `global` - Original algorithm, sorts by total count (larger corpora dominate)
- `stratified` - Selects top types from EACH corpus proportionally (balanced representation)

**Data Flow:**
```
Stage 4 (chunks) → Stage 4.5 autotune OR 4.6 → Stage 6b (Neo4j + Leiden) → Query
                         │                            │
                         ▼                            ▼
              extraction_results.json         communities.json
              (entities + relationships)      (Leiden clusters + summaries)
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
│   ├── indexing/                 # Stage 6: Vector DB (Weaviate)
│   ├── retrieval/                # Stage 7: Query -> Chunks
│   │   ├── preprocessing/        # Query transformation (strategies)
│   │   ├── reranking.py          # Cross-encoder
│   │   └── rrf.py                # RRF merging for decomposition/graphrag
│   └── generation/               # Stage 8: Chunks -> Answer
│
├── graph/                        # GraphRAG: Knowledge graph + communities
│   ├── schemas.py                # GraphEntity, Community models
│   ├── extractor.py              # LLM entity extraction
│   ├── neo4j_client.py           # Neo4j operations
│   ├── community.py              # Leiden + summarization
│   └── query.py                  # Hybrid graph retrieval
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
| `src/graph/extractor.py` | Entity extraction | `run_extraction(strategy) -> Dict` |
| `src/graph/neo4j_client.py` | Neo4j operations | `upload_extraction_results(driver, results) -> counts` |
| `src/graph/community.py` | Leiden communities | `detect_and_summarize_communities(driver, gds) -> List[Community]` |
| `src/graph/query.py` | Graph retrieval | `hybrid_graph_retrieval(query, results, driver) -> merged` |

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
- `evaluation-workflow.md` - Evaluation architecture and design decisions
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
- Available strategies now: `none`, `hyde`, `decomposition`, `graphrag`

**Phase 4: Query Decomposition** - COMPLETE
- [x] Add DECOMPOSITION_PROMPT and decompose_query() function
- [x] Implement decomposition_strategy in strategies.py
- [x] Register decomposition in STRATEGIES dict
- [x] Update config, CLI, logger, and UI

**Phase 5: Alpha Tuning + Comprehensive Eval** - COMPLETE (Dec 24-Jan 1)
- [x] Comprehensive evaluation mode: `--comprehensive` flag for 5D grid search
- [x] Tests all combinations: collections × search_types × alphas × strategies × top_k
- [x] Curated 15-question subset in `comprehensive_questions.json`
- [x] Leaderboard report with metric breakdowns
- [x] Trace persistence for metric recalculation (Dec 31)
- [x] Retrieval caching for top_k dimension (Dec 31)
- [x] Retry logic with exponential backoff (Dec 31)
- [x] Added search_type dimension (keyword vs hybrid) - Jan 1

**Phase 6: Contextual Retrieval** - COMPLETE (Dec 22)
- [x] Create contextual_chunker.py (prepend LLM context to chunks)
- [x] Add contextual_strategy to strategies.py
- [x] Run via: `python -m src.stages.run_stage_4_chunking --strategy contextual`

**Phase 7: RAPTOR** - COMPLETE (Dec 25)
- [x] Create raptor/ module (tree_builder.py, clustering.py, summarizer.py, schemas.py)
- [x] Add raptor_strategy to strategies.py
- [x] Run via: `python -m src.stages.run_stage_4_5_raptor`

**Phase 8: GraphRAG** - COMPLETE (Dec 25)
- [x] Add Neo4j to docker-compose.yml (with GDS plugin for Leiden)
- [x] Create src/graph/ module (schemas, extractor, neo4j_client, community, query)
- [x] Add Stage 4.6 (entity extraction) and Stage 6b (Neo4j upload + Leiden)
- [x] Add graphrag preprocessing strategy with hybrid retrieval (RRF merge)
- [x] Run via: `python -m src.stages.run_stage_4_6_graph_extract`
- [x] Upload: `python -m src.stages.run_stage_6b_neo4j`

**Note:** Evaluation runs via CLI (`python -m src.stages.run_stage_7_evaluation`), not in UI.

### Completed Recently
- Added search_type dimension: keyword (BM25) vs hybrid as 5th evaluation axis (Jan 1)
- Evaluation docs update: 5D grid, traces, caching, design decisions (Jan 1)
- Retrieval caching: top_k as innermost loop, 50% fewer API calls in grid search (Dec 31)
- Top_k as 4th evaluation dimension: [10, 20] to test retrieval depth impact (Dec 31)
- Trace schemas + retry logic: QuestionTrace, EvaluationTrace, exponential backoff (Dec 31)
