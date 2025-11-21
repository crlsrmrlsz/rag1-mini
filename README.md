# RAG1-Mini

A lightweight Retrieval-Augmented Generation (RAG) system combining cognitive neuroscience and philosophy to answer questions about human behavior.

## Project Goal

Build a specialized AI that integrates:
- **Cognitive Neuroscience** (David Eagleman)
- **Stoic Philosophy** (Marcus Aurelius)
- **Life Wisdom** (Schopenhauer, Gracián)

## Current Status

**Phase 1: PDF Text Extraction** - Testing and comparing extraction methods for multi-column academic documents.

## Project Structure

```
rag1-mini/
├── src/
│   ├── pdf_extractors/    # PDF extraction methods
│   ├── ingest.py          # Document ingestion (planned)
│   ├── chunk.py           # Text chunking (planned)
│   ├── embed.py           # Embedding generation (planned)
│   ├── vectorstore.py     # Vector storage (planned)
│   └── rag_server.py      # API server (planned)
├── data/
│   ├── raw/               # Source PDFs
│   └── debug/             # Debug visualizations
├── memory-bank/           # Project documentation
└── tests/                 # Test suite
```

## Environment

```bash
# Activate conda environment
conda activate rag1-mini

# Run PDF extraction tests
python src/pdf_extractors/pdf_extract_pymupdf_blocks.py
```

## Pipeline Phases

1. **PDF Extraction** (Current) - Extract clean text from academic PDFs
2. **Chunking** - Intelligent text segmentation
3. **Embedding** - Generate semantic vectors
4. **Vector Storage** - Index and store embeddings
5. **Retrieval** - Query and retrieve relevant context
6. **LLM Integration** - Generate grounded answers
7. **API Layer** - REST endpoint for queries

## Documentation

See `memory-bank/` for detailed project context and progress tracking.
