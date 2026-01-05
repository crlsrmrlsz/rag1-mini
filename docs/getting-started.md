# Getting Started

## About This Project

RAGLab is a **learning and showcase project** created to experiment with RAG (Retrieval-Augmented Generation) improvement techniques. I built this to understand how different chunking strategies, query preprocessing methods, and evaluation approaches affect RAG system performance.

**Important:** The data used in this project (19 books on neuroscience and philosophy) is **not included** in the repository due to intellectual property protection. The code is published for educational purposes.

You can use this project to:
- Learn how a complete RAG pipeline works
- Create your own dataset with your own documents
- Experiment with different RAG techniques

---

## Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for Weaviate and Neo4j)
- **OpenRouter API key** (for embeddings and LLM calls) - get one at [openrouter.ai](https://openrouter.ai)

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/raglab.git
cd raglab

# Create conda environment (recommended)
conda create -n raglab python=3.10
conda activate raglab

# Install dependencies
pip install -e .

# Install spaCy model (for sentence segmentation)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
# Or for general text: python -m spacy download en_core_web_sm

# Create .env file with your API key
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your_key_here

# Start Docker services
docker compose up -d
```

---

## Project Structure

```
src/
├── content_preparation/    # Stages 1-3: PDF → clean text
├── rag_pipeline/           # Stages 4-7: chunking, embedding, retrieval, generation
│   ├── chunking/           # Section, contextual, semantic chunkers
│   ├── embedding/          # OpenAI embedding API
│   ├── indexing/           # Weaviate upload
│   ├── retrieval/          # Search + preprocessing strategies
│   └── generation/         # Answer generation
├── graph/                  # GraphRAG: Neo4j, Leiden communities
├── evaluation/             # RAGAS evaluation framework
├── ui/                     # Streamlit application
└── stages/                 # CLI entry points for each stage

data/
├── raw/                    # Your source PDFs go here
│   └── {corpus}/           # Organize by topic (e.g., mybooks/)
└── processed/              # Pipeline outputs (auto-created)
```

---

## Data Folder Structure

When you run the pipeline, data flows through these folders:

| Folder | Created By | Contains |
|--------|------------|----------|
| `data/raw/{corpus}/` | You | Your source PDF files |
| `data/processed/01_raw_extraction/` | Stage 1 | Markdown extracted from PDFs |
| `data/processed/02_manual_review/` | (Optional) | Reviewed markdown before cleaning |
| `data/processed/03_markdown_cleaning/` | Stage 2 | Cleaned markdown files |
| `data/processed/04_nlp_chunks/` | Stage 3 | Sentence-segmented JSON |
| `data/processed/05_final_chunks/{strategy}/` | Stage 4 | Chunks ready for embedding |
| `data/processed/06_embeddings/{strategy}/` | Stage 5 | Chunks with vector embeddings |
| `data/processed/07_graph/` | Stage 4.5b | GraphRAG entities and relationships |

---

## Pipeline Stages

| Stage | Command | What It Does |
|-------|---------|--------------|
| 1 | `python -m src.stages.run_stage_1_extraction` | PDF → Markdown (using Docling) |
| 2 | `python -m src.stages.run_stage_2_processing` | Clean markdown (remove artifacts) |
| 3 | `python -m src.stages.run_stage_3_segmentation` | Sentence segmentation (spaCy NLP) |
| 4 | `python -m src.stages.run_stage_4_chunking` | Create chunks (800 tokens, 2-sentence overlap) |
| 4.5a | `python -m src.stages.run_stage_4_5_raptor` | RAPTOR hierarchical tree (optional) |
| 4.5b | `python -m src.stages.run_stage_4_5_autotune` | GraphRAG entity extraction (optional) |
| 5 | `python -m src.stages.run_stage_5_embedding` | Generate embeddings (OpenAI API) |
| 6 | `python -m src.stages.run_stage_6_weaviate` | Upload to Weaviate vector database |
| 6b | `python -m src.stages.run_stage_6b_neo4j` | Upload to Neo4j + Leiden communities (optional) |
| 7 | `python -m src.stages.run_stage_7_evaluation` | RAGAS evaluation |

**Note:** Stages 4.5a, 4.5b, and 6b are optional advanced techniques. The basic pipeline is stages 1-6.

---

## Running the UI

```bash
streamlit run src/ui/app.py
```

Open http://localhost:8501 in your browser. The UI allows you to:
- Select embedding collection and search parameters
- Choose preprocessing strategy (HyDE, decomposition, GraphRAG)
- Ask questions and see retrieved chunks, LLM interactions, and final answers
- Compare results across different configurations

---

## Quick Start

The fastest path to get a working RAG system with your own documents:

1. **Add your PDFs**
   ```bash
   mkdir -p data/raw/mybooks
   cp /path/to/your/*.pdf data/raw/mybooks/
   ```

2. **Run the pipeline** (stages 1-6)
   ```bash
   python -m src.stages.run_stage_1_extraction
   python -m src.stages.run_stage_2_processing
   python -m src.stages.run_stage_3_segmentation
   python -m src.stages.run_stage_4_chunking
   python -m src.stages.run_stage_5_embedding
   python -m src.stages.run_stage_6_weaviate
   ```

3. **Query your documents**
   ```bash
   streamlit run src/ui/app.py
   ```

For detailed documentation on each technique, see the [Documentation Index](README.md).
