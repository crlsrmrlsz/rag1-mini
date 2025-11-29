# RAG1-Mini

A lightweight Retrieval-Augmented Generation (RAG) system combining cognitive neuroscience and philosophy to answer questions about human behavior.

## Project Goal

Build a specialized AI that integrates:
- **Cognitive Neuroscience** (David Eagleman)
- **Stoic Philosophy** (Marcus Aurelius)
- **Life Wisdom** (Schopenhauer, Gracián)

## Current Status

**Phase 1: PDF Text Extraction** ✅ Completed - All source PDFs extracted using pymupdf4llm.
**Phase 2: Markdown Cleaning** 🔄 In Progress - Developing text cleaning pipeline.

## Project Structure

```
rag1-mini/
├── src/text_extractor/
│   └── pdf_extract.py    # PDF → Markdown with fallback (pymupdf4llm)
├── data/
│   ├── raw/              # Source PDFs (neuroscience, wisdom)
│   └── processed/        # Extracted markdown files
├── memory-bank/          # Project documentation
├── notebooks/            # Extraction method explorations
└── tests/
```

## Environment

```bash
# Activate conda environment
conda activate rag1-mini
```

## Usage

Extract all PDFs from `data/raw/` to `data/processed/`:
```bash
python src/text_extractor/pdf_extract.py
```

Features:
- Whole-document extraction using `pymupdf4llm.to_markdown()` (layout-aware)
- Fallback to page-by-page processing for difficult documents
- OCR support for image-heavy pages
- Preserves subdirectory structure

## Pipeline Phases

1. **PDF Extraction** ✅ Completed - Extract clean layout-aware markdown from academic PDFs
2. **Markdown Cleaning** 🔄 Current - Remove artifacts, normalize formatting
3. **Chunking** - Intelligent text segmentation
4. **Embedding** - Generate semantic vectors
5. **Vector Storage** - Index and store embeddings
6. **Retrieval** - Query and retrieve relevant context
7. **LLM Integration** - Generate grounded answers
8. **API Layer** - REST endpoint for queries

## Documentation

See `memory-bank/` for detailed project context and progress tracking.
