# 🧠 RAG1 Mini - Neuro-Philosophy RAG System

## Project Overview
RAG1 Mini is a targeted RAG system that creates a hybrid **expert neuroscientist + philosopher** for answering questions about **human behavior**. The system integrates:
- Cognitive neuroscience (David Eagleman)
- Stoic philosophy (Marcus Aurelius)
- Life wisdom (Arthur Schopenhauer, Baltasar Gracián)

The learning objective is mastering RAG pipeline components while building a specialized AI that provides grounded, thoughtful answers about human cognition and behavior.

## Core Requirements

### Phase 1 — Data Ingestion & Preprocessing
- Extract text from PDF books using PyMuPDF with heuristics for headers/footers, multi-column pages, captions, and block merging
- Convert to clean paragraph-level JSONL format (`{text, page}`)
- Manual validation of extraction quality

### Phase 2 — Chunking & Embedding
- Intelligent chunking (250–350 tokens, 15–20% overlap)
- Local embeddings using BGE-base-en or E5-base-v2 (CPU-friendly)
- Vector storage in Chroma or FAISS

### Phase 3 — Retrieval Pipeline
- Query embedding using same model
- Top-k retrieval and optional cross-encoder reranking
- Structured retrieval results with semantic preservation

### Phase 4 — Local LLM Orchestration
- Quantized LLM inference (Llama-2 7B Q4_K_M, Mistral 7B Q4, Phi-3-mini)
- Citation-based prompt templates requiring JSON output
- Grounded answer generation

### Phase 5 — API Layer
- FastAPI REST endpoint: `POST /ask {"query": "Why do humans procrastinate?"}`
- Full pipeline orchestration from embed → retrieve → LLM → response

### Phase 6 — Evaluation
- 20–30 test questions for quality validation
- Context recall and factuality evaluation
- Optional RAGAS and Phoenix (Arize) integration

## Success Criteria
- Accurate PDF text extraction with reading order preservation
- Effective chunking maintaining semantic coherence across neuroscience-philosophy contexts
- High-quality embeddings enabling relevant cross-domain retrieval
- Thoughtful, evidence-based answers about human behavior
- < 2-second API response time
- Clean, documented code with proper citations

## Scope Limitations
- **Content**: 1–2 Eagleman chapters + 1 philosophical text (keeping scope manageable)
- **Deployment**: Partially local (embeddings/vector DB) to expose production constraints
- **Technology**: Python-only implementation
- **Architecture**: Single-user REST API, no UI components
- **Vector Storage**: Chroma or FAISS (simplicity over SQLite)

## Key Deliverables
- Complete neuro-philosophy RAG pipeline
- `clean_paragraphs.jsonl` from data processing
- `vector_store/` with embedded content
- `retriever.py` for query processing
- `inference.py` with LLM orchestration
- FastAPI `main.py` with `/ask` endpoint
- `eval/` directory with test suite and metrics
