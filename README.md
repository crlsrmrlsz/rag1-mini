# RAG1-Mini

A lightweight Retrieval-Augmented Generation (RAG) system combining cognitive neuroscience and philosophy to answer questions about human behavior.

## Project Goal

Build a specialized AI that integrates:
- **Cognitive Neuroscience** (David Eagleman, Robert Sapolsky, John Pinel, etc.)
- **Stoic Philosophy** (Marcus Aurelius, Epictetus, Seneca)
- **Life Wisdom** (Schopenhauer, Gracián, Confucius, Lao Tzu, Daniel Kahneman)

## Current Status

**Stage 1: PDF Text Extraction** ✅ **COMPLETE**
**Stage 2: Processing and Cleaning** ✅ **COMPLETE**  
**Stage 3: NLP Segmentation & Chunking** ✅ **COMPLETE**

**Stage 4+: RAG Implementation** 🔄 **Ready to Begin**

All 19 books have been successfully processed through the complete extraction → cleaning → chunking pipeline. The system is now ready for vector embedding and RAG implementation.

## Project Structure

```
rag1-mini/
├── src/
│   ├── run_stage_1_extraction.py     # Stage 1: PDF → Markdown
│   ├── run_stage_2_processing.py     # Stage 2: Cleaning & Review Processing
│   ├── run_stage_3_segmentation.py   # Stage 3: NLP Chunking
│   ├── config.py                     # Configuration settings
│   ├── extractors/
│   │   └── docling_parser.py         # PDF extraction logic
│   └── processors/
│       ├── text_cleaner.py           # Markdown cleaning
│       └── nlp_segmenter.py          # NLP chunking
├── data/
│   ├── raw/                          # Original PDFs (19 files)
│   └── processed/
│       ├── 01_raw_extraction/        # Stage 1 output (19 MD files)
│       ├── 02_manual_review/         # Stage 2 manual review
│       ├── 03_markdown_cleaning/     # Stage 2 cleaned files
│       └── 04_final_chunks/          # Stage 3 output (38 files: JSON + MD)
├── memory-bank/
│   ├── progress.md                   # Detailed project progress
│   ├── activeContext.md              # Current system state
│   └── projectbrief.md               # Project overview
├── notebooks/                        # Jupyter notebooks for analysis
└── logs/
    └── cleaning_report.log           # Processing logs
```

## Environment

```bash
# Activate conda environment
conda activate rag1-mini
```

## Usage

The processing pipeline consists of 3 completed stages, with a future RAG implementation phase.

### Completed Stages

#### Stage 1: PDF Extraction ✅
Extracts text from all PDFs in `data/raw/` and saves them as Markdown files.
```bash
python -m src.run_stage_1_extraction
```

#### Stage 2: Processing and Cleaning ✅
Takes manually reviewed files, cleans them, and standardizes the format.
```bash
python -m src.run_stage_2_processing
```

#### Stage 3: NLP Segmentation & Chunking ✅
Performs advanced NLP processing and creates semantic chunks with metadata.
```bash
python -m src.run_stage_3_segmentation
```

### Current Outputs

**Stage 3 Results in `data/processed/04_final_chunks/`:**
- **38 files total:** 19 JSON files + 19 Markdown files
- **Neuroscience:** 8 books (psychology, cognitive science, brain research)
- **Wisdom:** 11 books (philosophy, stoicism, behavioral economics)
- **Metadata:** Each chunk includes book_name, category, chunk_id, source info
- **Format:** Both structured JSON (for processing) and readable Markdown (for humans)

## Pipeline Phases

1. **PDF Extraction** ✅ **COMPLETE** - Extract clean, layout-aware markdown from 19 PDFs
2. **Manual Review** ✅ **COMPLETE** - Manual cleaning and verification of extracted content
3. **Processing and Chunking** ✅ **COMPLETE** - NLP-based segmentation with metadata
4. **Embedding** 🔄 **Ready** - Generate semantic vectors for each chunk
5. **Vector Storage** 🔄 **Ready** - Index and store embeddings in vector database
6. **Retrieval** 🔄 **Ready** - Query and retrieve relevant context
7. **LLM Integration** 🔄 **Ready** - Generate grounded answers
8. **API Layer** 🔄 **Ready** - REST endpoint for queries

## Content Categories

### Neuroscience (8 books)
- **Robert Sapolsky:** "Behave" & "Determined"
- **John Pinel & Steven Barnes:** "Biopsychology" 
- **David Eagleman & Jonathan Downar:** "Brain and Behavior"
- **Michael Gazzaniga:** "Cognitive Neuroscience"
- **Luca Tommasi et al.:** "Cognitive Biology"
- **Nicole Gage & Bernard:** "Fundamentals of Cognitive Neuroscience"
- **Konstanthos Fountoulakis & Ioannis Nimatoudis:** "Psychobiology of Behaviour"

### Wisdom & Philosophy (11 books)
- **Daniel Kahneman:** "Thinking, Fast and Slow"
- **Arthur Schopenhauer:** Multiple works (Essays, Counsels and Maxims, Wisdom of Life)
- **Marcus Aurelius:** "Meditations"
- **Epictetus:** "The Enchiridion" & "The Art of Living"
- **Seneca:** "Letters from a Stoic"
- **Confucius:** "The Analects"
- **Lao Tzu:** "Tao Te Ching"
- **Baltasar Gracian:** "The Pocket Oracle and Art of Prudence"

## Next Steps

The project is now ready for **Stage 4: RAG Implementation**. The processed chunks are high-quality and ready for:

1. **Vector Embedding Generation** - Using local or API-based embedding models
2. **Vector Database Setup** - Chroma, Pinecone, or similar vector store
3. **Retrieval System** - Semantic search and context retrieval
4. **LLM Integration** - Grounded response generation
5. **API Development** - RESTful interface for queries

## Documentation

See `memory-bank/` for detailed project context and progress tracking:
- `progress.md` - Comprehensive project progress and statistics
- `activeContext.md` - Current system state and readiness assessment
- `projectbrief.md` - Original project goals and requirements
