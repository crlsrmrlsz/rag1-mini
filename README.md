# RAG1 Mini - Retrieval-Augmented Generation System

A lightweight implementation of a RAG system with local document processing and retrieval capabilities.

## Project Structure

```
rag1-mini/
│
├── data/
│   ├── eagleman_ch1.txt
│   ├── philosophy1.txt
│
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── vectorstore.py
│   ├── rag_server.py
│   ├── model/
│       ├── llama.cpp  (optional)
│
└── tests/
    └── test_questions.json
```

## Components

- **ingest.py**: Document ingestion and preprocessing
- **chunk.py**: Text chunking utilities
- **embed.py**: Text embedding generation
- **vectorstore.py**: Vector storage and retrieval
- **rag_server.py**: API server for RAG queries
- **model/llama.cpp**: Optional local LLM integration

## Usage

```bash
python src/ingest.py
python src/rag_server.py
