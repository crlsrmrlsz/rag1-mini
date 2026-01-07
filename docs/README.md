# RAGLab Documentation

Technical documentation for the RAGLab RAG experimentation pipeline.

## Getting Started

New here? Start with **[Getting Started](getting-started.md)** for installation, project structure, and how to run the pipeline with your own data.

## Documentation Sections

| Section | What You'll Find |
|---------|------------------|
| [Content Preparation](content-preparation/content-preparation.md) | PDF extraction with Docling, markdown cleaning, and NLP sentence segmentation (Stages 1-3) |
| [Chunking Strategies](chunking/) | Document splitting approaches: section-based, contextual retrieval, and RAPTOR hierarchical summarization (Stage 4) |
| [Preprocessing Strategies](preprocessing/) | Query-time transformations: HyDE, query decomposition, GraphRAG, and cross-encoder reranking |
| [Evaluation Framework](evaluation/) | RAGAS metrics, comprehensive grid search across configurations, and evaluation results |

## Quick Commands

```bash
# Basic pipeline (stages 1-6)
python -m src.stages.run_stage_1_extraction    # PDF → Markdown
python -m src.stages.run_stage_2_processing    # Clean markdown
python -m src.stages.run_stage_3_segmentation  # Sentence segmentation
python -m src.stages.run_stage_4_chunking      # Create chunks
python -m src.stages.run_stage_5_embedding     # Generate embeddings
python -m src.stages.run_stage_6_weaviate      # Upload to Weaviate

# RAPTOR (hierarchical summarization)
python -m src.stages.run_stage_4_5_raptor      # Build summary tree

# GraphRAG (knowledge graph + communities)
python -m src.stages.run_stage_4_5_autotune    # Extract entities
python -m src.stages.run_stage_6b_neo4j        # Upload to Neo4j + Leiden

# Launch the UI
streamlit run src/ui/app.py
```

