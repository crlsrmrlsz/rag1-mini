# RAGLab

An advanced Retrieval-Augmented Generation pipeline implementing 6 modern techniques from 2024-2025 research papers. Built to deeply understand RAG concepts—not just use a framework.

## Techniques Implemented

| Technique | Paper | What It Does |
|-----------|-------|--------------|
| **HyDE** | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | Generates hypothetical answers for semantic matching |
| **Query Decomposition** | [arXiv:2507.00355](https://arxiv.org/abs/2507.00355) | Breaks complex questions into sub-queries with RRF merging |
| **Contextual Chunking** | [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval) | LLM-generated context prepended to chunks (-35% retrieval failures) |
| **RAPTOR** | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) | Hierarchical summarization tree with UMAP + GMM clustering |
| **GraphRAG** | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) | Knowledge graph + Leiden communities for cross-document reasoning |
| **GraphRAG Auto-Tuning** | [MS Research](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/) | Discovers entity types from corpus content (per-book resumable) |

Plus: Hybrid search (BM25 + vector), cross-encoder reranking, structured LLM outputs, and RAGAS evaluation framework.

## Documentation

For implementation details, design decisions, and code walkthroughs:

- **[Chunking Strategies](docs/chunking/)** — Section, Contextual, RAPTOR
- **[Preprocessing Strategies](docs/preprocessing/)** — HyDE, Decomposition, GraphRAG
- **[Evaluation Framework](docs/evaluation/)** — RAGAS metrics and results

Each technique includes paper references, key code snippets, and tradeoff analysis.

## What I Learned

Building this pipeline taught me that RAG is deceptively complex. A few key insights:

**PDF parsing is harder than expected.** Scientific books with complex layouts, figures, and footnotes break naive extraction. Docling helped, but significant cleaning was still needed.

**Prompts make or break LLM-based techniques.** HyDE, RAPTOR summarization, and entity extraction all depend heavily on prompt engineering. Small wording changes dramatically affect output quality.

**Evaluation is the hardest part.** Generating good test questions for RAGAS requires domain expertise. The gap between "looks reasonable" and "measurably good" is where real learning happens.

**GraphRAG complexity is justified.** The knowledge graph + Leiden communities approach seemed over-engineered at first, but it handles cross-document reasoning that vector search alone cannot.

## Corpus & Evaluation

- **19 books** (8 neuroscience + 11 philosophy)
- **45 custom test questions** with expert-written reference answers
- **RAGAS metrics**: Faithfulness, relevancy, context precision, answer correctness (F1)
- **Grid search evaluation** across all strategy combinations

## Highlights

- **Full 8-stage pipeline**: PDF extraction to answer generation
- **Multiple chunking strategies**: Section-based, contextual, RAPTOR hierarchical
- **Multiple preprocessing strategies**: None, HyDE, decomposition, GraphRAG
- **Clean architecture**: Function-based design with fail-fast error handling
- **~7,700 lines** of Python code (no framework wrappers)

## Technologies

| Category | Tools |
|----------|-------|
| **Vector Database** | Weaviate (HNSW + BM25 hybrid) |
| **Graph Database** | Neo4j (GDS plugin for Leiden communities) |
| **LLM API** | OpenRouter (GPT-4, Claude, embeddings) |
| **NLP** | spaCy (en_core_sci_sm), tiktoken |
| **PDF Processing** | Docling |
| **Data Validation** | Pydantic (structured LLM outputs) |
| **UI** | Streamlit |
| **Evaluation** | RAGAS framework |
| **Infrastructure** | Docker, Conda |

## Quick Start

```bash
# Setup environment
conda activate raglab

# Run pipeline stages (baseline)
python -m src.stages.run_stage_1_extraction   # Extract PDFs
python -m src.stages.run_stage_2_processing   # Clean markdown
python -m src.stages.run_stage_3_segmentation # NLP segmentation
python -m src.stages.run_stage_4_chunking     # Create chunks
python -m src.stages.run_stage_5_embedding    # Generate embeddings
python -m src.stages.run_stage_6_weaviate     # Upload to Weaviate

# Advanced: RAPTOR hierarchical summarization
python -m src.stages.run_stage_4_5_raptor     # Build summary tree

# Advanced: GraphRAG knowledge graph
python -m src.stages.run_stage_4_5_autotune       # Auto-discover entity types
python -m src.stages.run_stage_4_6_graph_extract  # Extract entities
python -m src.stages.run_stage_6b_neo4j           # Upload to Neo4j

# Launch search UI
docker compose up -d                          # Start Weaviate + Neo4j
streamlit run src/ui/app.py                   # Open http://localhost:8501

# Run evaluation
python -m src.stages.run_stage_7_evaluation   # RAGAS quality metrics
python -m src.stages.run_stage_7_evaluation --comprehensive  # Grid search
```

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

Each strategy evaluated as black box (questions in, metrics out). See [evaluation-workflow.md](memory-bank/evaluation-workflow.md) for diagrams.

## Requirements

- Python 3.8+
- Conda environment: `raglab`
- OpenRouter API key
- Docker (for Weaviate + Neo4j)

## License

MIT
