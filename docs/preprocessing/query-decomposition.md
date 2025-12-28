# Query Decomposition

> **Paper:** [Question Decomposition for Retrieval-Augmented Generation](https://arxiv.org/abs/2507.00355) | Ammann et al. (Humboldt-Universität) | July 2025

Breaks complex multi-part questions into simpler sub-questions, retrieves for each, then merges results using Reciprocal Rank Fusion (RRF).

## TL;DR

For complex queries like "Compare X and Y", decomposition generates sub-queries like "What is X?", "What is Y?", "How do X and Y differ?". Each sub-query retrieves independently, and RRF merges the results. Paper reports **+36.7% MRR@10** on multi-hop benchmarks.

## The Problem

Complex questions often require multiple pieces of information:

```
Query: "How does Stoic philosophy's view on emotions compare to
        modern neuroscience findings about emotional regulation?"
```

This requires:
1. What Stoics said about emotions
2. What neuroscience says about emotional regulation
3. How they relate

A single retrieval struggles because no single chunk contains all three aspects.

## The Solution

### Decomposition Flow

```
Original Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM DECOMPOSITION                                          │
│                                                             │
│  → "What is the Stoic view on emotions?"                    │
│  → "What does neuroscience say about emotional regulation?" │
│  → "What similarities or differences exist between..."      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PARALLEL RETRIEVAL                                         │
│                                                             │
│  Sub-query 1 → [results_1]                                  │
│  Sub-query 2 → [results_2]                                  │
│  Sub-query 3 → [results_3]                                  │
│  Original    → [results_0]  (always retained!)              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  RRF MERGE                                                  │
│                                                             │
│  score(doc) = sum(1 / (60 + rank_i)) for each query_i      │
│  → Documents appearing in multiple result lists boosted     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Merged, Ranked Chunks
```

### Critical Design: Original Query Retained

The paper emphasizes: **always include the original query** in retrieval. This:
- Preserves overall context
- Acts as fallback for simple queries
- Catches chunks relevant to the whole question

## Implementation Details

### Decomposition Prompt

```python
# src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py

DECOMPOSITION_PROMPT = """Break down this question for a knowledge base on cognitive science and philosophy.

If the question is simple enough to answer directly, keep it as a single question.
Otherwise, create 3-5 sub-questions that can be answered independently and together cover all aspects of the original.

Question: {query}

Respond with JSON:
{{
  "sub_questions": ["...", "...", "..."],
  "reasoning": "Brief explanation"
}}"""
```

**Key design choices:**
- **"If simple, keep as single question"**: Haystack best practice — avoid over-decomposition
- **"Answered independently"**: EfficientRAG finding — enables parallel retrieval
- **"Cover all aspects"**: LangChain approach — improves recall
- **3-5 sub-questions**: Paper uses max 5

### Strategy Function

```python
# src/rag_pipeline/retrieval/preprocessing/strategies.py

def decomposition_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Decompose query into sub-questions for RRF merging."""
    model = model or PREPROCESSING_MODEL

    # Decompose query
    sub_queries, reasoning = decompose_query(query, model=model)

    # Build query list: original + sub-queries
    generated_queries = [{"type": "original", "query": query}]
    for i, sq in enumerate(sub_queries):
        generated_queries.append({"type": f"sub_{i+1}", "query": sq})

    return PreprocessedQuery(
        original_query=query,
        search_query=query,  # Keep original for display
        sub_queries=sub_queries,
        generated_queries=generated_queries,  # For RRF
        strategy_used="decomposition",
    )
```

### RRF Merging

```python
# src/rag_pipeline/retrieval/rrf.py

def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    query_types: List[str],
    k: int = 60,
    top_k: int = 10,
) -> RRFResult:
    """Merge multiple result lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = defaultdict(float)
    results_by_id: Dict[str, SearchResult] = {}

    for query_idx, results in enumerate(result_lists):
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id

            # RRF formula: 1 / (k + rank + 1)
            rrf_score = 1.0 / (k + rank + 1)
            scores[chunk_id] += rrf_score

            # Keep highest original score version
            if chunk_id not in results_by_id or result.score > results_by_id[chunk_id].score:
                results_by_id[chunk_id] = result

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return RRFResult(results=[...], ...)
```

**Why RRF?**
- No score normalization needed (ranks are comparable)
- Documents in multiple lists get boosted
- `k=60` is standard (from Cormack et al., 2009)

### Design Decisions

**Temperature 0.7 (paper uses 0.8)?**
- Paper uses 0.8 + nucleus sampling for diversity
- 0.7 balances diversity with coherence
- Higher temperatures sometimes produce nonsensical sub-queries

**Why JSON output format?**
- Structured output enables easy parsing
- Reasoning field helps debug prompt issues
- Fallback to original query if parsing fails

**Why 3-5 sub-questions (not fixed 5)?**
- Simple queries don't need decomposition
- "Keep as single question" clause handles this
- Flexibility matches question complexity

## When to Use

**Good for:**
- Comparison questions ("How does X compare to Y?")
- Multi-aspect questions ("What, how, and why...?")
- Questions spanning multiple topics or time periods
- Complex multi-hop reasoning

**Limitations:**
- Overhead for simple factual queries
- Sub-query quality depends on LLM
- Multiple retrievals add latency

## Example

**Query**: "How do Stoic techniques for managing anger compare to what neuroscience tells us about emotional regulation?"

**Decomposition**:
```json
{
  "sub_questions": [
    "What techniques did the Stoics use for managing anger?",
    "What does neuroscience research reveal about emotional regulation mechanisms?",
    "Are there overlaps between ancient Stoic practices and modern neuroscience findings?",
    "What are the key differences between Stoic approaches and neuroscience-based techniques?"
  ],
  "reasoning": "Comparison question requiring both Stoic philosophy content and neuroscience content"
}
```

**Retrieval**: Each sub-query finds relevant chunks; RRF merges with boost for chunks appearing in multiple results.

## Cost Analysis

- **Decomposition call**: ~100 input + ~150 output tokens
- **Per query cost**: ~$0.0001 (gpt-4o-mini)
- **Retrieval**: 4-6 searches instead of 1 (parallelizable)

Latency dominated by LLM call (~500ms), not by extra retrievals.

## Results

See [Evaluation Results](../evaluation/results.md) for RAGAS metrics comparing decomposition against none, HyDE, and GraphRAG.

## Related

- [HyDE](hyde.md) — Alternative for semantic matching
- [GraphRAG](graphrag.md) — Entity-based alternative
- [RRF Implementation](../../src/rag_pipeline/retrieval/rrf.py) — Merge algorithm
