# RAG1-Mini

A production-quality Retrieval-Augmented Generation (RAG) pipeline that creates a specialized AI combining cognitive neuroscience with philosophical wisdom to answer questions about human behavior.

## Highlights

- **19 Books Processed**: 8 neuroscience texts + 11 philosophy/wisdom books
- **6,245 Semantic Chunks**: Section-aware chunking with sentence overlap
- **7-Stage Pipeline**: Extraction, Cleaning, Segmentation, Chunking, Embedding, Vector Storage, Search UI
- **Clean Architecture**: Function-based design with fail-fast error handling

## Pipeline Overview

```
PDF Files (19)
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
```

## Quick Start

```bash
# Setup environment
conda activate rag1-mini

# Run pipeline stages
python -m src.run_stage_1_extraction   # Extract PDFs
python -m src.run_stage_2_processing   # Clean markdown
python -m src.run_stage_3_segmentation # NLP segmentation
python -m src.run_stage_4_chunking     # Create chunks
python -m src.run_stage_5_embedding    # Generate embeddings
python -m src.run_stage_6_weaviate     # Upload to Weaviate

# Launch search UI
docker compose up -d                   # Start Weaviate
streamlit run src/ui/app.py            # Open http://localhost:8501
```

## Project Structure

```
rag1-mini/
├── src/
│   ├── run_stage_1_extraction.py    # PDF to Markdown
│   ├── run_stage_2_processing.py    # Markdown cleaning
│   ├── run_stage_3_segmentation.py  # NLP segmentation
│   ├── run_stage_4_chunking.py      # Section chunking
│   ├── run_stage_5_embedding.py     # Embedding generation
│   ├── config.py                    # Central configuration
│   ├── extractors/
│   │   └── docling_parser.py        # PDF extraction
│   ├── processors/
│   │   ├── text_cleaner.py          # Markdown cleaning
│   │   └── nlp_segmenter.py         # Sentence segmentation
│   ├── ingest/
│   │   ├── naive_chunker.py         # Token-aware chunking
│   │   └── embed_texts.py           # Embedding API client
│   ├── vector_db/
│   │   ├── weaviate_client.py       # Weaviate connection & upload
│   │   └── weaviate_query.py        # Search functions
│   ├── ui/
│   │   └── app.py                   # Streamlit search interface
│   └── utils/
│       ├── file_utils.py            # File operations
│       └── tokens.py                # Token counting
├── data/
│   ├── raw/                         # Original PDFs (19 files)
│   └── processed/
│       ├── 01_raw_extraction/       # Stage 1 output
│       ├── 02_manual_review/        # Manual review
│       ├── 03_markdown_cleaning/    # Stage 2 output
│       ├── 04_nlp_chunks/           # Stage 3 output
│       ├── 05_final_chunks/         # Stage 4 output (6,245 chunks)
│       └── 06_embeddings/           # Stage 5 output
└── memory-bank/                     # Project documentation
```

## Content Library

### Neuroscience (8 books)

| Author | Title |
|--------|-------|
| Robert Sapolsky | Behave, Determined |
| David Eagleman & Jonathan Downar | Brain and Behavior |
| John Pinel & Steven Barnes | Biopsychology |
| Michael Gazzaniga | Cognitive Neuroscience |
| Luca Tommasi et al. | Cognitive Biology |
| Nicole Gage & Bernard | Fundamentals of Cognitive Neuroscience |
| Fountoulakis & Nimatoudis | Psychobiology of Behaviour |

### Philosophy & Wisdom (11 books)

| Author | Title |
|--------|-------|
| Daniel Kahneman | Thinking, Fast and Slow |
| Marcus Aurelius | Meditations |
| Epictetus | The Enchiridion, The Art of Living |
| Seneca | Letters from a Stoic |
| Arthur Schopenhauer | Essays, Counsels and Maxims, Wisdom of Life |
| Confucius | The Analects |
| Lao Tzu | Tao Te Ching |
| Baltasar Gracian | The Pocket Oracle and Art of Prudence |

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

## Requirements

- Python 3.8+
- Conda environment: `rag1-mini`
- Dependencies: docling, spacy, tiktoken, requests, weaviate-client, streamlit
- OpenRouter API key (for embeddings)
- Docker (for Weaviate vector database)

## License

MIT
