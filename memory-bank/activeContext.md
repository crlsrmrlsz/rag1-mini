# RAG1 Mini - Neuro-Philosophy Active Context

## Current Work Focus

**Phase 1 (Data Ingestion)**: Currently implementing and refining PDF text extraction with PyMuPDF. Focus on multi-column detection, reading order preservation, and clean paragraph extraction from Eagleman neuroscience and philosophical texts.

**Architecture Implementation**: Building towards the integrated neuro-philosophy RAG system that combines cognitive neuroscience (Eagleman) with Stoic philosophy insights.

## Active Decisions and Considerations

### Content Selection
- **Eagleman Chapters**: Using 1–2 chapters from "Incognito" or similar neuroscience texts
- **Philosophical Texts**: Stoic philosophy (Marcus Aurelius) plus life wisdom (Schopenhauer, Gracián)
- **Integration Focus**: Cross-referencing neuroscience findings with philosophical perspectives

### Technical Stack Choices
- **Embedding Models**: Evaluating BGE-base-en vs E5-base-v2 for CPU-friendly performance
- **Vector Storage**: Chroma vs FAISS (Chroma likely preferred for simpler Python integration)
- **LLM Selection**: Phi-3-mini, Mistral 7B Q4, or Llama-2 7B Q4_K_M for local inference
- **PDF Processing**: PyMuPDF with enhanced heuristics for academic document structures

### Architecture Pipeline

```mermaid
flowchart TD

    A[PDF Books] --> B[PyMuPDF Extractor]
    B --> C[Clean Paragraph JSONL]

    C --> D[Chunker]
    D --> E[Embedding Model (BGE/E5)]

    E --> F[Chroma/FAISS Vector Store]

    G[User Query] --> H[Embed Query]
    H --> F
    F --> I[Top-k Context]

    I --> J[Prompt Builder]
    J --> K[Local LLM (llama.cpp / Ollama)]

    K --> L[FastAPI Output (JSON)]
```

### Immediate Priorities
1. **Complete Phase 1**: Robust PDF extraction with validation
2. **Establish Chunking Strategy**: 250–350 tokens with 15–20% overlap
3. **Implement Embedding Pipeline**: Local model selection and optimization
4. **Build Retrieval Component**: Top-k with optional reranking
5. **Create LLM Integration**: Citation-based prompts and JSON outputs
6. **Develop API Endpoint**: `POST /ask` with full pipeline orchestration

## Important Patterns and Preferences

### Coding Principles
- **DRY/KISS**: Avoid duplication, prefer simple solutions over complexity
- **Type Hints**: Use Python typing for clarity and IDE support
- **Function Design**: Small, focused functions with clear responsibilities
- **Citation Requirements**: All AI outputs must include text references

### Data Pipeline Philosophy
- **Phased Processing**: Raw PDFs → Clean JSONL → Chunks → Embeddings → Store
- **Validation Points**: Manual checks at each phase boundary
- **Error Transparency**: Clear error messages for troubleshooting
- **Reproducibility**: Deterministic processing for consistent results

### Testing Strategy
- **Content Validation**: Manual review of extraction and chunking quality
- **Retrieval Testing**: 20–30 test questions covering neuro-philosophy topics
- **Integration Tests**: Full pipeline end-to-end validation
- **Performance Benchmarks**: Response time < 2 seconds target

## Current Known Issues
- **PDF Layout Challenges**: Multi-column academic texts need refined extraction heuristics
- **Cross-Domain Matching**: Ensuring neuroscience and philosophy contexts align semantically
- **Citation Accuracy**: Maintaining source attribution through processing pipeline
- **Memory Constraints**: Local LLM inference within reasonable RAM limits

## Project Insights
- **Educational Value**: Learning production constraints through local deployment requirements
- **Domain Specialization**: Shows power of focused AI vs general-purpose systems
- **Integration Challenge**: Cross-disciplinary knowledge synthesis requires careful prompt engineering
- **Performance Trade-offs**: CPU-only local processing teaches optimization fundamentals
- **Evidence Grounding**: Citation requirements build trust and AI accountability
