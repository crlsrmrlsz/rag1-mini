# RAG1 Mini - Neuro-Philosophy System Patterns

## System Architecture

### Neuro-Philosophy Pipeline
```
Raw PDFs → PyMuPDF Extraction → Clean JSONL → Chunker → Local Embeddings → Vector Store → Cross-Encoder Reranking → LLM Prompting → Citation-Based Answer → JSON API Response
```

The pipeline creates a specialized AI that integrates cognitive neuroscience (Eagleman) with philosophical wisdom (Stoics, Schopenhauer, Gracián) for thoughtful answers about human behavior.

## Key Technical Decisions

### Component Specialization
- **PDF Extractor**: PyMuPDF with multi-column detection and reading order preservation
- **Chunker**: 250–350 token segments with 15–20% overlap, maintaining semantic coherence
- **Embedder**: BGE-base-en or E5-base-v2 for CPU-friendly cross-domain similarity
- **Vector Store**: Chroma or FAISS for efficient neuro-philosophy context retrieval
- **Retriever**: Top-k selection with optional cross-encoder reranking
- **LLM Orchestrator**: Citation-based prompts requiring structured JSON outputs
- **API Layer**: FastAPI `/ask` endpoint with < 2 second performance target

### Cross-Domain Integration Patterns
- **Citation Grounding**: Every answer must reference specific text sources
- **Context Preservation**: Maintain neuroscience-philosophy source attribution
- **Query Embedding**: Use same model for consistent semantic space
- **Integration Verification**: Manual validation of interdisciplinary coherence

## Design Patterns in Use

### Strategy Pattern
- **Multiple Embedding Models**: BGE vs E5 evaluation for best neuro-philosophy matching
- **Flexible Vector Stores**: ChromaFAISS implementations for performance comparison
- **Modular LLMs**: Phi-3-mini, Mistral, Llama-2 options for capability testing

### Repository Pattern
- **Knowledge Base**: Abstracts storage of neuroscience and philosophical texts
- **Citation Tracking**: Maintains source attribution through retrieval pipeline
- **Cross-Reference Support**: Enables neuroscience philosophy linking

### Pipeline Pattern
- **Phase Isolation**: Each of 6 phases (Data → Chunk → Embed → Retrieve → LLM → API)
- **Progress Checkpointing**: JSONL outputs enable resumable processing
- **Error Containment**: Component failures don't cascade beyond phase boundaries

## Critical Implementation Paths

### Data Ingestion Flow
1. **Input**: Eagleman neuroscience + philosophy PDFs
2. **PyMuPDF Processing**: Multi-column detection, header/footer removal, caption handling
3. **Paragraph Extraction**: Reading order preservation, block merging heuristics
4. **Output**: `clean_paragraphs.jsonl` ({text, page, source} format)
5. **Validation**: Manual review of extraction quality

### Chunking & Embedding Flow
1. **Input**: Clean paragraph JSONL
2. **Chunking**: 250–350 tokens with 15–20% overlap, semantic boundary respect
3. **Parallel Embedding**: Batch processing with BGE-base-en or E5-base-v2
4. **Output**: Embedded chunks with source attribution
5. **Storage**: Chroma/FAISS vector database with metadata

### Retrieval & Synthesis Flow
1. **Query Input**: Human behavior questions like "Why do humans procrastinate?"
2. **Query Embedding**: Same model as document embedding for semantic matching
3. **Vector Search**: Top-k retrieval from neuroscience-philosophy knowledge base
4. **Optional Reranking**: Cross-encoder for improved relevance
5. **LLM Synthesis**: Citation-based prompt requiring evidence-grounded answers

## Component Relationships

```
FastAPI (/ask endpoint)
            ↓
Retriever (embed → search → rerank)
            ↓
Vector Store (Chroma/FAISS)
            ↓
Embedder (BGE/E5) ← Chunker
            ↓          ↑
PDF Extractor ←─────── LLM Orchestrator
   ↑                       ↓
Raw PDFs           Citation-Prompt Templates
```

- Components communicate via structured data formats (JSONL, embeddings, vectors)
- Each component focuses on single responsibility with clear interfaces
- Pipeline enables both batch processing and live query processing

## Citation & Grounding Patterns
- **Source Attribution**: Always include (text, page, author) references
- **Cross-Domain Links**: Connect neuroscience findings with philosophical insights
- **Confidence Scoring**: Include retrieval relevance scores in responses
- **Answer Transparency**: JSON output showing reasoning process

## Performance Patterns
- **CPU Optimization**: Embedding models selected for workstation performance
- **Batch Processing**: Parallel embedding generation and indexing
- **Memory Management**: Streaming for large document sets
- **Response Targets**: < 2 second end-to-end query processing

## Testing Patterns
- **Domain Validation**: 20–30 test questions covering neuro-philosophy integration
- **Citation Verification**: Manual checking of source attribution accuracy
- **Cross-Domain Testing**: Questions requiring both neuroscience and philosophy
- **Performance Benchmarking**: Response time and resource usage measurement
- **Integration Tests**: End-to-end pipeline validation with real questions
