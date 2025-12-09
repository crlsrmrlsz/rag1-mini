# RAG1-Mini Project Brief

## Overview
RAG1-Mini is a specialized RAG system that creates a **hybrid neuroscientist + philosopher** for answering questions about human behavior. It integrates cognitive neuroscience (David Eagleman) with Stoic philosophy (Marcus Aurelius) and life wisdom (Schopenhauer, Gracián).

## Core Goal
Master RAG pipeline components while building a specialized AI that provides grounded, thoughtful answers about human cognition and behavior.

## Why This Exists
- **Cross-domain synthesis**: Bridges neuroscience with philosophical wisdom
- **Evidence-based insights**: All answers grounded in source citations
- **Production learning**: Exposes real deployment constraints (performance, memory, accuracy)
- **Specialized AI**: Demonstrates domain expertise beyond generic assistants

## System Architecture

```
Raw PDFs -> Stage 1: PyMuPDF Extraction -> Manual Review -> Stage 2: Processing & Chunking ->
Local Embeddings -> Vector Store -> Retrieval -> LLM -> Citation-Based Answer -> JSON API Response
```

## Pipeline Phases

### Phase 1: PDF Extraction (✅ Completed)
- **Script**: `src/run_stage_1_extraction.py`
- Extracts text from PDFs into layout-aware markdown files.
- Output is stored in `data/processed/01_raw_extraction/`.

### Phase 2: Processing and Chunking (🔄 In Progress)
- **Manual Review**: Extracted markdown is manually cleaned and moved to `data/processed/02_manual_review/`.
- **Script**: `src/run_stage_2_processing.py`
- **Actions**:
    - Performs structural cleaning on the markdown.
    - Segments the text into intelligent chunks using a semantic segmenter.
    - Enriches chunks with metadata, including book name, chapter, and section.
- **Output**: Processed chunks are saved in `data/processed/04_final_chunks/`.

### Phase 3: Chunking & Embedding
- Intelligent chunking (250-350 tokens, 15-20% overlap).
- Local embeddings using BGE-base-en or E5-base-v2.
- Vector storage in Chroma or FAISS.

### Phase 4: Retrieval Pipeline
- Query embedding using same model.
- Top-k retrieval with optional cross-encoder reranking.
- Maintain source attribution.

### Phase 5: LLM Integration
- Quantized LLM (Llama-2 7B, Mistral 7B, or Phi-3-mini).
- Citation-based prompts requiring JSON output.
- Grounded answer generation.

### Phase 6: API Layer
- FastAPI REST endpoint: `POST /ask`.
- Full pipeline orchestration.
- < 2 second response time target.

### Phase 7: Evaluation
- 20-30 test questions about human behavior.
- Context recall and factuality metrics.
- Optional RAGAS integration.

## Technical Stack

- **Python 3.8+** with conda environment `rag1-mini`
- **PyMuPDF**: PDF text extraction
- **SciSpacy**: NLP sentence segmentation
- **sentence-transformers**: Local embeddings (BGE/E5)
- **Chroma/FAISS**: Vector storage
- **FastAPI**: REST API
- **llama.cpp/Ollama**: Local LLM inference

## Success Criteria

- Accurate PDF extraction with reading order preservation.
- High-quality chunks with meaningful metadata.
- 80% of answers integrate neuroscience + philosophy appropriately.
- All claims backed by specific text citations.
- < 2 second API response time.
- Clean, documented code.

## Scope Limitations

- **Content**: 1-2 Eagleman chapters + 1 philosophical text
- **Deployment**: Local-only (CPU-friendly)
- **Architecture**: Single-user REST API, no UI
- **Technology**: Python-only implementation
