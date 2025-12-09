# RAG1-Mini

A lightweight Retrieval-Augmented Generation (RAG) system combining cognitive neuroscience and philosophy to answer questions about human behavior.

## Project Goal

Build a specialized AI that integrates:
- **Cognitive Neuroscience** (David Eagleman)
- **Stoic Philosophy** (Marcus Aurelius)
- **Life Wisdom** (Schopenhauer, Gracián)

## Current Status

**Phase 1: PDF Text Extraction** ✅ Completed.
**Phase 2: Processing and Chunking** 🔄 In Progress.

## Project Structure

```
rag1-mini/
├── src/
│   ├── run_stage_1_extraction.py   # Stage 1: PDF -> Markdown
│   └── run_stage_2_processing.py   # Stage 2: Markdown -> Chunks
├── data/
│   ├── raw/
│   └── processed/
│       ├── 01_raw_extraction/      # Stage 1 output
│       ├── 02_manual_review/       # Stage 2 input
│       ├── 03_structural_debug/    # Stage 2 debug output
│       └── 04_final_chunks/        # Stage 2 final output
├── memory-bank/
└── notebooks/
```

## Environment

```bash
# Activate conda environment
conda activate rag1-mini
```

## Usage

The processing pipeline is divided into two stages, with a manual review step in between.

### Stage 1: PDF Extraction

This script extracts text from all PDFs in `data/raw/` and saves them as Markdown files in `data/processed/01_raw_extraction/`. Run it from the project root directory.

```bash
python -m src.run_stage_1_extraction
```

### Manual Review

After Stage 1 is complete, manually inspect the generated Markdown files in `data/processed/01_raw_extraction/`. Correct any extraction errors or formatting issues. Once a file is reviewed and ready for processing, **move it** to the `data/processed/02_manual_review/` directory.

### Stage 2: Processing and Chunking

This script takes the manually reviewed Markdown files from `data/processed/02_manual_review/`, cleans them, segments them into chunks, and adds metadata (including the book name). The final output is saved in `data/processed/04_final_chunks/` in both JSON and Markdown formats. Run it from the project root directory.

```bash
python -m src.run_stage_2_processing
```

## Pipeline Phases

1.  **PDF Extraction** ✅ Completed - Extract clean, layout-aware markdown from PDFs.
2.  **Manual Review** 🔄 Current - Manually clean and verify extracted markdown.
3.  **Processing and Chunking** 🔄 Current - Clean, segment, and add metadata to the text.
4.  **Embedding** - Generate semantic vectors for each chunk.
5.  **Vector Storage** - Index and store embeddings.
6.  **Retrieval** - Query and retrieve relevant context.
7.  **LLM Integration** - Generate grounded answers.
8.  **API Layer** - REST endpoint for queries.

## Documentation

See `memory-bank/` for detailed project context and progress tracking.