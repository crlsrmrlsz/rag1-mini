# Architecture

RAGLab processes PDF documents through an 8-stage pipeline with multiple strategy options at chunking and query time.

## Pipeline Overview

```
PDF Documents
     |
     v
+-----------------------+
|  Stage 1: Extract     |  Docling PDF extraction
+-----------------------+
     |
     v
+-----------------------+
|  Stage 2: Clean       |  Regex-based artifact removal
+-----------------------+
     |
     v
+-----------------------+
|  Stage 3: Segment     |  spaCy NLP sentence segmentation
+-----------------------+
     |
     v
+-----------------------+
|  Stage 4: Chunk       |  800-token chunks, 2-sentence overlap
+-----------------------+
     |
     v
+-----------------------+
|  Stage 5: Embed       |  OpenRouter API embeddings
+-----------------------+
     |
     v
+-----------------------+
|  Stage 6: Store       |  Weaviate vector database
+-----------------------+
     |
     v
+-----------------------+
|  Stage 7: Search UI   |  Streamlit interface
+-----------------------+
     |
     v
+-----------------------+
|  Stage 8: Evaluate    |  RAGAS quality metrics
+-----------------------+
```

## Evaluation Architecture

Two independent axes for A/B testing:

```
CHUNKING STRATEGIES (Index Time)
├── section    → Baseline (800 tokens, 2-sentence overlap)
├── contextual → LLM context prepended to each chunk
└── raptor     → Hierarchical summary tree

PREPROCESSING STRATEGIES (Query Time)
├── none          → Original query unchanged
├── hyde          → Hypothetical passage replaces query
├── decomposition → Sub-queries + RRF merge
└── graphrag      → Entity extraction + Neo4j traversal
```

Each strategy evaluated as black box (questions in, metrics out). See [evaluation workflow](../memory-bank/evaluation-workflow.md) for diagrams.

## Project Structure

```
raglab/
├── src/
│   ├── content_preparation/         # Phase 1: Documents -> Text
│   │   ├── extraction/              # Stage 1: PDF -> Markdown
│   │   ├── cleaning/                # Stage 2: Clean Markdown
│   │   └── segmentation/            # Stage 3: Sentence splits
│   │
│   ├── rag_pipeline/                # Phase 2: RAG System
│   │   ├── chunking/                # Stage 4: Text -> Chunks
│   │   │   ├── section_chunker.py   # Baseline chunking
│   │   │   ├── contextual_chunker.py # Anthropic-style context
│   │   │   └── raptor/              # RAPTOR hierarchical tree
│   │   ├── embedding/               # Stage 5: Chunks -> Vectors
│   │   ├── indexing/                # Stage 6: Weaviate + Neo4j
│   │   ├── retrieval/               # Stage 7: Query -> Chunks
│   │   │   ├── preprocessing/       # HyDE, decomposition, graphrag
│   │   │   ├── reranking.py         # Cross-encoder
│   │   │   └── rrf.py               # Multi-query fusion
│   │   └── generation/              # Stage 8: Chunks -> Answer
│   │
│   ├── graph/                       # GraphRAG module
│   │   ├── auto_tuning.py           # Entity type discovery
│   │   ├── extractor.py             # Entity extraction
│   │   ├── neo4j_client.py          # Neo4j operations
│   │   ├── community.py             # Leiden communities
│   │   └── query.py                 # Hybrid graph+vector retrieval
│   │
│   ├── evaluation/                  # RAGAS framework
│   ├── ui/                          # Streamlit app
│   ├── shared/                      # Common utilities
│   ├── stages/                      # Pipeline stage runners
│   └── config.py                    # Central configuration
│
├── data/
│   ├── raw/                         # Original PDF documents
│   └── processed/                   # Stage outputs (01-06)
│
└── memory-bank/                     # Project documentation
```

## Key Modules

| Module | Purpose | Interface |
|--------|---------|-----------|
| `content_preparation/extraction/docling_parser.py` | PDF extraction | `extract_pdf(path) -> str` |
| `content_preparation/cleaning/text_cleaner.py` | Markdown cleaning | `run_structural_cleaning(text, name) -> (str, log)` |
| `content_preparation/segmentation/nlp_segmenter.py` | Sentence segmentation | `segment_document(text, name) -> List[Dict]` |
| `rag_pipeline/chunking/section_chunker.py` | Section chunking | `run_section_chunking() -> Dict[str, int]` |
| `rag_pipeline/embedding/embedder.py` | Embedding API | `embed_texts(texts) -> List[List[float]]` |
| `rag_pipeline/indexing/weaviate_client.py` | Weaviate storage | `upload_embeddings(client, name, chunks) -> int` |
| `rag_pipeline/retrieval/preprocessing/` | Query preprocessing | `preprocess_query(query) -> PreprocessedQuery` |
| `rag_pipeline/generation/answer_generator.py` | Answer synthesis | `generate_answer(query, chunks) -> GeneratedAnswer` |
| `graph/extractor.py` | Entity extraction | `run_extraction(strategy) -> Dict` |
| `graph/community.py` | Leiden communities | `detect_and_summarize_communities(driver, gds) -> List[Community]` |
