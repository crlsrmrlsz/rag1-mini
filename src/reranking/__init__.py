"""Reranking module for RAG1-Mini.

This module provides cross-encoder reranking to improve retrieval quality.

## Why Reranking Matters

Your retrieval pipeline uses a **bi-encoder** (embedding model) that encodes
queries and documents separately, then compares them with cosine similarity.
This is fast but loses fine-grained semantic relationships.

A **cross-encoder** processes query and document TOGETHER through a transformer,
enabling it to understand subtle relationships that bi-encoders miss.

## Two-Stage Retrieval Architecture

```
User Query
    ↓
Stage 1: Bi-Encoder (Fast, Approximate)
    - Embed query and documents separately
    - Retrieve top-50 candidates via vector search
    - O(1) for each candidate (pre-computed embeddings)
    ↓
Stage 2: Cross-Encoder (Slow, Precise)
    - Process each [query, document] pair together
    - Score relevance with full transformer attention
    - O(n) where n = number of candidates (e.g., 50)
    ↓
Final Results: Top-10 Reranked
```

## Usage

```python
from src.reranking import rerank

# Get initial candidates from hybrid search
candidates = query_hybrid(client, query, top_k=50)

# Rerank to top-10 with cross-encoder
reranked = rerank(query, candidates, top_k=10)
```
"""

from src.reranking.cross_encoder import rerank, get_reranker, RerankResult

__all__ = ["rerank", "get_reranker", "RerankResult"]
