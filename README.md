# RAG1-Mini

A production-quality Retrieval-Augmented Generation (RAG) pipeline designed for learning and experimentation. Process any document collection through an 8-stage pipeline to build a searchable knowledge base with AI-powered answers.

## Highlights

- **Full RAG Pipeline**: 8 stages from PDF extraction to answer generation
- **Section-Aware Chunking**: Intelligent text segmentation with overlap
- **Advanced Retrieval**: Hybrid search, query preprocessing, cross-encoder reranking
- **Clean Architecture**: Function-based design with fail-fast error handling

## Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Vector Database** | Weaviate (HNSW + BM25 hybrid) |
| **LLM API** | OpenRouter (GPT-4, Claude, embeddings) |
| **NLP** | spaCy (en_core_sci_sm), tiktoken |
| **PDF Processing** | Docling |
| **Data Validation** | Pydantic (structured LLM outputs) |
| **UI** | Streamlit |
| **Evaluation** | RAGAS framework |
| **Infrastructure** | Docker, Conda |

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

## Quick Start

```bash
# Setup environment
conda activate rag1-mini

# Run pipeline stages
python -m src.stages.run_stage_1_extraction   # Extract PDFs
python -m src.stages.run_stage_2_processing   # Clean markdown
python -m src.stages.run_stage_3_segmentation # NLP segmentation
python -m src.stages.run_stage_4_chunking     # Create chunks
python -m src.stages.run_stage_5_embedding    # Generate embeddings
python -m src.stages.run_stage_6_weaviate     # Upload to Weaviate

# Launch search UI
docker compose up -d                          # Start Weaviate
streamlit run src/ui/app.py                   # Open http://localhost:8501

# Run evaluation
python -m src.stages.run_stage_7_evaluation   # RAGAS quality metrics
```

## Project Structure

The codebase is organized into two main phases for learners:

```
RAG1-Mini: A Teaching RAG Pipeline
══════════════════════════════════

📚 CONTENT PREPARATION (Stages 1-3)
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ Extraction  │ →  │  Cleaning   │ →  │Segmentation │
   │ (PDF→MD)    │    │ (Fix MD)    │    │ (Sentences) │
   └─────────────┘    └─────────────┘    └─────────────┘
         ↓
🤖 RAG PIPELINE (Stages 4-8)
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  Chunking   │ →  │  Embedding  │ →  │  Indexing   │
   │ (Sections)  │    │ (Vectors)   │    │ (Weaviate)  │
   └─────────────┘    └─────────────┘    └─────────────┘
         ↓
   ┌─────────────────────────────────────────────────────┐
   │              RETRIEVAL (Stage 7)                    │
   │  Query → Preprocess → Search → Rerank → Generate   │
   └─────────────────────────────────────────────────────┘
```

### Directory Layout

```
rag1-mini/
├── src/
│   ├── content_preparation/         # Phase 1: Documents → Text
│   │   ├── extraction/              # Stage 1: PDF → Markdown
│   │   │   └── docling_parser.py
│   │   ├── cleaning/                # Stage 2: Clean Markdown
│   │   │   └── text_cleaner.py
│   │   └── segmentation/            # Stage 3: Sentence splits
│   │       └── nlp_segmenter.py
│   │
│   ├── rag_pipeline/                # Phase 2: RAG System
│   │   ├── chunking/                # Stage 4: Text → Chunks
│   │   │   └── section_chunker.py
│   │   ├── embedding/               # Stage 5: Chunks → Vectors
│   │   │   └── embedder.py
│   │   ├── indexing/                # Stage 6: Vector DB
│   │   │   ├── weaviate_client.py
│   │   │   └── weaviate_query.py
│   │   ├── retrieval/               # Stage 7: Query → Chunks
│   │   │   ├── preprocessing/       # Query transformation
│   │   │   ├── reranking.py         # Cross-encoder
│   │   │   └── rrf.py               # Multi-query fusion
│   │   └── generation/              # Stage 8: Chunks → Answer
│   │       └── answer_generator.py
│   │
│   ├── evaluation/                  # RAGAS framework
│   │   └── ragas_evaluator.py
│   ├── ui/                          # Streamlit app
│   │   └── app.py
│   ├── shared/                      # Common utilities
│   │   ├── openrouter_client.py     # Unified LLM API
│   │   ├── files.py
│   │   └── tokens.py
│   ├── stages/                      # Pipeline stage runners
│   │   ├── run_stage_1_extraction.py
│   │   ├── run_stage_2_processing.py
│   │   └── ...
│   └── config.py                    # Central configuration
│
├── data/
│   ├── raw/                         # Original PDF documents
│   └── processed/
│       ├── 01_raw_extraction/       # Stage 1 output
│       ├── 02_manual_review/        # Manual review
│       ├── 03_markdown_cleaning/    # Stage 2 output
│       ├── 04_nlp_chunks/           # Stage 3 output
│       ├── 05_final_chunks/         # Stage 4 output
│       └── 06_embeddings/           # Stage 5 output
└── memory-bank/                     # Project documentation
```

## Technical Design

### Architecture Principles

- **Function-based**: Functions as primary interface; classes only for stateful components
- **Fail-fast**: Exceptions propagate immediately; no log-and-continue
- **Absolute imports**: Consistent `from src.module import ...` pattern
- **Centralized config**: All settings in `src/config.py`

### Chunking Strategy

- **Token limit**: 800 tokens per chunk (optimized for embedding models)
- **Overlap**: 2 sentences carried between consecutive chunks
- **Section-aware**: Respects document structure boundaries
- **Quality filtering**: Removes sentence fragments and artifacts

### Key Parameters

```python
MAX_CHUNK_TOKENS = 800      # Target chunk size
OVERLAP_SENTENCES = 2       # Context continuity
SPACY_MODEL = "en_core_sci_sm"  # Scientific NLP
```

## RAG Techniques Applied

This project implements advanced RAG patterns from recent research:

| Technique | Description | Paper/Source |
|-----------|-------------|--------------|
| **Hybrid Search** | BM25 keyword + vector semantic search | Weaviate docs |
| **Step-Back Prompting** | Abstracts questions to broader concepts (+27% on multi-hop) | [arXiv:2310.06117](https://arxiv.org/abs/2310.06117) |
| **Multi-Query + RRF** | Generates targeted queries, merges with Reciprocal Rank Fusion | [Query Decomposition](https://arxiv.org/html/2507.00355v1) |
| **Query Decomposition** | Breaks complex questions into sub-queries (+36.7% MRR@10) | [Haystack Blog](https://haystack.deepset.ai/blog/query-decomposition) |
| **Cross-Encoder Reranking** | Re-scores results with BERT (+20-35% precision) | sentence-transformers |
| **Structured LLM Outputs** | Pydantic + JSON Schema enforcement | OpenAI structured outputs |
| **Section-Aware Chunking** | Respects document boundaries with overlap | RAG best practices |
| **RAGAS Evaluation** | LLM-as-judge via LangChain wrapper (faithfulness, relevancy, context precision) | [RAGAS framework](https://docs.ragas.io/) |

## Requirements

- Python 3.8+
- Conda environment: `rag1-mini`
- Dependencies: docling, spacy, tiktoken, requests, weaviate-client, streamlit, ragas, langchain-openai
- OpenRouter API key (for embeddings)
- Docker (for Weaviate vector database)

## License

MIT
