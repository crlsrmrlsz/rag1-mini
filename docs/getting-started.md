# Getting Started

## Prerequisites

- Python 3.8+
- Conda environment: `raglab`
- OpenRouter API key (set in environment)
- Docker (for Weaviate + Neo4j)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/raglab.git
cd raglab

# Create conda environment
conda create -n raglab python=3.10
conda activate raglab
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY=your_key_here
```

## Quick Start

```bash
# Start databases
docker compose up -d

# Launch UI
streamlit run src/ui/app.py
# Open http://localhost:8501
```

## Full Pipeline

### Stage 1-6: Baseline Pipeline

```bash
python -m src.stages.run_stage_1_extraction   # PDF -> Markdown
python -m src.stages.run_stage_2_processing   # Clean Markdown
python -m src.stages.run_stage_3_segmentation # NLP sentence segmentation
python -m src.stages.run_stage_4_chunking     # 800-token chunks
python -m src.stages.run_stage_5_embedding    # Generate embeddings
python -m src.stages.run_stage_6_weaviate     # Upload to Weaviate
```

### Stage 4.5: RAPTOR (Hierarchical Summaries)

```bash
python -m src.stages.run_stage_4_5_raptor     # Build summary tree
```

Creates hierarchical clusters with LLM-generated summaries using UMAP + GMM clustering.

### Stage 4.5-4.6: GraphRAG (Knowledge Graph)

```bash
# Option A: Auto-discover entity types from corpus
python -m src.stages.run_stage_4_5_autotune --strategy section

# Option B: Use predefined entity types
python -m src.stages.run_stage_4_6_graph_extract --strategy section

# Upload to Neo4j + run Leiden community detection
python -m src.stages.run_stage_6b_neo4j
```

## Evaluation

```bash
# Single configuration
python -m src.stages.run_stage_7_evaluation \
  --collection RAG_section_embed3large_v1 \
  --preprocessing hyde \
  --alpha 0.7

# Grid search (all combinations)
python -m src.stages.run_stage_7_evaluation --comprehensive
```

Results appended to `memory-bank/evaluation-history.md`.

## Chunking Strategies

| Strategy | Command | Description |
|----------|---------|-------------|
| `section` | `--strategy section` | 800-token baseline with 2-sentence overlap |
| `contextual` | `--strategy contextual` | LLM context prepended to each chunk |
| `raptor` | `run_stage_4_5_raptor` | Hierarchical summary tree |

## Preprocessing Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| `none` | `--preprocessing none` | Original query unchanged |
| `hyde` | `--preprocessing hyde` | Hypothetical document embeddings |
| `decomposition` | `--preprocessing decomposition` | Sub-queries with RRF merge |
| `graphrag` | `--preprocessing graphrag` | Entity extraction + Neo4j traversal |

## Configuration

Key settings in `src/config.py`:

```python
MAX_CHUNK_TOKENS = 800          # Target chunk size
OVERLAP_SENTENCES = 2           # Sentence overlap between chunks
WEAVIATE_HOST = "localhost"     # Vector database host
WEAVIATE_HTTP_PORT = 8080       # REST API port
WEAVIATE_GRPC_PORT = 50051      # gRPC port (v4 client)
```

## Troubleshooting

**Weaviate connection error:**
```bash
docker compose up -d
# Wait 10 seconds for startup
```

**Neo4j not responding:**
```bash
docker compose logs neo4j
# Check GDS plugin is loaded
```

**Out of memory:**
- Reduce `MAX_CHUNK_TOKENS` in config
- Process fewer books at once
