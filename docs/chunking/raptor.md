# RAPTOR: Hierarchical Summarization

> **Paper:** [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) | Sarthi et al. (Stanford/Google) | ICLR 2024

Builds a hierarchical tree of summaries from document chunks, enabling retrieval at multiple levels of abstraction. Answers both "What did Sapolsky say about cortisol?" and "What is the main argument of this book?"

## TL;DR

RAPTOR recursively clusters chunks using UMAP + GMM, generates LLM summaries for each cluster, then clusters the summaries. The result is a multi-level tree where:
- **Leaves** (level 0): Original document chunks
- **Summaries** (levels 1+): Progressively more abstract representations

At query time, all nodes (leaves + summaries) are searched together — the query naturally retrieves the right level of abstraction.

## Key Results (Paper)

- **+20% absolute accuracy** on QuALITY benchmark (complex reasoning)
- **55.7% F1** on QASPER (new SOTA vs 53.0% for DPR)
- **18.5-57%** of retrieved nodes come from summary layers

## The Problem

Traditional RAG retrieves only leaf-level chunks. This fails for:
- **Theme questions**: "What is the author's central argument?"
- **Multi-section synthesis**: "How do chapters 3 and 7 connect?"
- **Comparative questions**: "What's the difference between X and Y approaches?"

These require information scattered across many chunks — no single chunk contains the answer.

## The Solution

### Tree Structure

```
+------------------------------------------------------------------+
|                     RAPTOR Tree Structure                         |
+------------------------------------------------------------------+
|                                                                   |
|  Level 3 (Root):      [    Document Summary    ]                  |
|                              ^                                    |
|  Level 2 (Summaries):  [S1]    [S2]    [S3]                       |
|                         ^       ^       ^                         |
|  Level 1 (Clusters):  +-+-+   +-+-+   +-+-+                       |
|                       |   |   |   |   |   |                       |
|  Level 0 (Leaves):   [C1][C2][C3][C4][C5][C6][C7]...              |
|                       ^   ^   ^   ^   ^   ^   ^                   |
|                    Original Document Chunks                       |
+------------------------------------------------------------------+
```

### Algorithm

```
For each book:
  1. Load section chunks as level-0 nodes (leaves)
  2. Embed all nodes

  While nodes.count > MIN_CLUSTER_SIZE:
    3. UMAP: Reduce embedding dimensions (1536 → 10)
    4. GMM: Find optimal K clusters using BIC
    5. GMM: Soft-cluster nodes (allows multi-cluster membership)
    6. For each cluster:
       - Concatenate member texts
       - Generate LLM summary
       - Create new node at level+1
    7. Embed new summary nodes
    8. Repeat with summary nodes as input

  Return all nodes (leaves + all summary levels)
```

## Implementation Details

### UMAP Dimensionality Reduction

```python
# src/rag_pipeline/chunking/raptor/clustering.py

from umap import UMAP

def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """Reduce high-dim embeddings for GMM clustering."""
    reducer = UMAP(
        n_neighbors=10,      # Balance local/global structure
        n_components=10,     # Target dimensions for GMM
        min_dist=0.0,        # Tight clusters
        metric='cosine'      # Match embedding similarity
    )
    return reducer.fit_transform(embeddings)
```

**Why UMAP?**
- GMM struggles with high dimensions (curse of dimensionality)
- UMAP preserves both local and global structure
- 10 dimensions is a sweet spot for clustering quality

### GMM Soft Clustering

```python
from sklearn.mixture import GaussianMixture

def find_optimal_clusters(embeddings: np.ndarray, max_k: int = 50) -> int:
    """Find optimal K using Bayesian Information Criterion."""
    bics = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(embeddings)
        bics.append(gmm.bic(embeddings))
    return np.argmin(bics) + 2  # Lower BIC is better
```

**Why GMM over K-means?**
- **Soft clustering**: A chunk about "stress and cortisol" can belong to both "neuroscience" and "health" clusters
- **Probabilistic**: Captures cluster shape and overlap
- **BIC optimization**: Data-driven cluster count (no magic numbers)

### LLM Summarization

```python
# src/rag_pipeline/chunking/raptor/summarizer.py

RAPTOR_SUMMARY_PROMPT = """Write a comprehensive summary of the following passages.
Include key details, names, and specific concepts.

Passages:
{context}

Summary:"""

def generate_cluster_summary(
    chunks: List[Dict],
    model: str = "anthropic/claude-3-haiku"
) -> str:
    """Generate summary for a cluster of chunks."""
    context = "\n\n".join(chunk["text"] for chunk in chunks)

    return call_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,
        max_tokens=200
    )
```

### Tree Building Orchestration

```python
# src/rag_pipeline/chunking/raptor/tree_builder.py

def build_raptor_tree(
    chunks: List[Dict],
    book_id: str,
    max_levels: int = 4,
    min_cluster_size: int = 3,
) -> Tuple[List[RaptorNode], TreeMetadata]:
    """Build hierarchical RAPTOR tree from section chunks."""

    # Convert chunks to RaptorNodes (level 0)
    nodes = [RaptorNode.from_chunk(c, level=0) for c in chunks]

    all_nodes = list(nodes)
    current_level = 0

    while len(nodes) >= min_cluster_size and current_level < max_levels:
        # Embed current level
        embeddings = embed_texts([n.text for n in nodes])

        # Cluster
        reduced = reduce_dimensions(embeddings)
        k = find_optimal_clusters(reduced)
        clusters = cluster_nodes(reduced, k)

        # Summarize each cluster
        summary_nodes = []
        for cluster_id, member_indices in clusters.items():
            members = [nodes[i] for i in member_indices]
            summary_text = generate_cluster_summary(members)

            summary_node = RaptorNode(
                chunk_id=f"{book_id}::L{current_level+1}_cluster_{cluster_id}",
                text=summary_text,
                tree_level=current_level + 1,
                is_summary=True,
                child_ids=[m.chunk_id for m in members],
                # ...
            )
            summary_nodes.append(summary_node)

        all_nodes.extend(summary_nodes)
        nodes = summary_nodes
        current_level += 1

    return all_nodes, TreeMetadata(...)
```

### Collapsed Tree Retrieval

At query time, RAPTOR uses "collapsed tree" retrieval (paper finding: outperforms tree traversal):

```python
# All nodes (leaves + summaries) are in the same Weaviate collection
# Query searches ALL nodes by similarity
# Higher-level summaries naturally match abstract queries
# Leaf nodes naturally match specific factual queries

results = weaviate_client.query_hybrid(
    collection="RAG_raptor_embed3large_v1",
    query_text=query,
    limit=20,
    alpha=0.7  # Favor vector over BM25
)
```

The magic: same embedding model encodes both chunks and summaries, so similarity naturally selects the right abstraction level.

## Design Decisions

**Why 800-token leaves (not paper's 100)?**
- RAGLab already optimized section chunks at 800 tokens
- Reduces tree depth (fewer LLM summarization calls)
- Maintains consistency with other strategies

**Why Claude-3-Haiku for summaries?**
- Fast and cheap (~$0.40 for full 19-book corpus)
- Sufficient quality for summarization task
- Configurable via `RAPTOR_SUMMARY_MODEL`

**Why per-book trees (not cross-book)?**
- Simpler implementation matching existing structure
- Cross-book would require major schema changes
- 19 books is small enough for within-book retrieval to work

## When to Use

**Good for:**
- Questions about themes and arguments
- Multi-section synthesis questions
- "What is this book about?"
- Comparison across chapters/sections

**Limitations:**
- Higher indexing cost (many LLM calls)
- Larger index size (leaves + summaries)
- Summary quality depends on LLM capability
- Overkill for simple factual queries

## Cost Analysis

For 19 books with ~150 chunks each:
- **Summaries**: ~36 per book × 19 = ~684 LLM calls
- **Cost**: ~$0.40 total (claude-3-haiku)
- **Time**: ~3 minutes per book (LLM-dominated)

## Results

See [Evaluation Results](../evaluation/results.md) for RAGAS metrics comparing RAPTOR against section and contextual chunking.

## Related

- [Section Chunking](section-chunking.md) — Prerequisite (RAPTOR uses section chunks as leaves)
- [Contextual Chunking](contextual-chunking.md) — Alternative approach (can be combined)
- [GraphRAG](../preprocessing/graphrag.md) — Different hierarchy via knowledge graphs
