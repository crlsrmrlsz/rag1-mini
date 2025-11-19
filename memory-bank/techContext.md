# RAG1 Mini - Tech Context

## Technologies Used

### Core Runtime
- **Python 3.8+**: Primary implementation language

### Document Processing
- **PyMuPDF (fitz/pymupdf)**: PDF parsing and text extraction
  - Multi-column layout detection for academic documents
  - Reading order preservation and block merging heuristics
  - Header/footer/caption identification and handling

### Machine Learning
- **Embedding Models**: BGE-base-en or E5-base-v2 (via sentence-transformers)
  - CPU-optimized for workstation deployment
  - Cross-domain semantic similarity (neuroscience ↔ philosophy)
  - Local inference without external API dependencies

### Vector Storage & Search
- **Chroma** or **FAISS**: Vector database for embedding storage
  - Efficient similarity search for neuro-philosophy contexts
  - Metadata preservation with source attribution
  - Python-native integration for simplicity

### Data Processing
- **NumPy**: Vector operations and mathematical computations
- **scikit-learn**: Optional cross-encoder reranking
  - Machine learning utilities for retrieval optimization

### Large Language Models
- **Local LLM Options**: Phi-3-mini, Mistral 7B Q4, or Llama-2 7B Q4_K_M
  - llama.cpp or Ollama for CPU inference
  - Quantized models for workstation performance
  - Citation-based prompt engineering

### Web Framework
- **FastAPI**: REST API for neuro-philosophy queries
  - `POST /ask` endpoint with JSON request/response
  - Type validation with Pydantic models
  - Asynchronous support for concurrency

## Development Setup

### Conda Environment
- Environment name: `rag1-mini`
- Isolated Python environment via conda/miniconda
- Activation: `conda activate rag1-mini`

### Project Structure
```
rag1-mini/
├── .clinerules/          # Contextual coding instructions
├── src/                  # Core pipeline components
├── data/                 # PDF sources and processed data
│   ├── raw/              # Original PDFs (Eagleman, philosophy)
│   ├── processed/        # Extracted text and metadata
│   └── debug/            # Visualization aids
├── vector_store/         # Chroma/FAISS databases
├── eval/                 # Test questions and metrics
├── notebooks/            # Jupyter exploration
└── memory-bank/          # Session continuity documentation
```

### Installation Dependencies
Core package requirements for neuro-philosophy pipeline:
- pymupdf>=1.23.0 (PDF extraction)
- sentence-transformers>=2.2.0 (embeddings)
- chromadb>=0.4.0 or faiss-cpu>=1.7.0 (vector storage)
- fastapi>=0.104.0 (API framework)
- uvicorn>=0.24.0 (ASGI server)
- llama-cpp-python>=0.2.0 or ollama>=0.1.0 (LLM inference)
- numpy>=1.24.0 (numerical computing)
- scikit-learn>=1.3.0 (optional reranking)

## Technical Constraints

### Resource Requirements
- **RAM**: 4GB minimum, 8GB recommended for larger documents
- **Storage**: 100MB base + 10x document size for processing
- **CPU**: Multi-core preferred for batch processing
- **GPU**: Optional (not required for basic functionality)

### Compatibility
- **OS**: Linux/macOS/Windows (via conda portability)
- **Python**: 3.8-3.11 (conda managed versions)
- **File Formats**: PDF-only (extensible to other formats)

### Limitations
- **Vector Ops**: CPU-only cosine similarity (SQLite-vec)
- **Scale**: Single-user, single-machine deployment
- **Models**: Pre-trained embeddings, download required on first use

## Development Workflows

### Environment Activation
```bash
# Always activate before Python execution
conda activate rag1-mini

# Check active environment
conda env list
```

### Testing Framework
- **pytest**: Unit test execution
- **JSON fixtures**: Test questions in `tests/test_questions.json`
- **Manual testing**: API calls via curl or browser

### Code Quality
- **black/isort**: Code formatting (if configured)
- **mypy**: Type checking (recommended)
- **pre-commit hooks**: Automatic quality checks

### Debugging
- **Jupyter notebooks**: Interactive exploration in `notebooks/`
- **Debug PDFs**: Visual markers for PDF layout analysis
- **Component isolation**: Test individual modules independently

## Build and Deployment

### Local Development
```bash
# Installation (assumed conda environment pre-configured)
pip install -r requirements.txt  # If exists

# Run pipeline
python src/ingest.py
python src/rag_server.py
```

### Deployment Strategy
- **Containerization**: Docker support available (detected in CLI tools)
- **Local-first**: Designed for workstation deployment
- **Zero-config**: No external services required
