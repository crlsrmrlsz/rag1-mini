# GraphRAG Implementation Analysis & Fix Plan

**Date:** 2026-01-07
**Updated:** 2026-01-08
**Status:** All Critical Issues Fixed
**Related:** [graphrag.md](graphrag.md), [graphrag-tutorial.md](graphrag-tutorial.md)

---

## Executive Summary

Deep analysis of GraphRAG implementation revealed several significant issues. All **critical issues have been fixed** as of 2026-01-08. The implementation now properly uses RRF merging, fetches graph-only chunks, normalizes entities consistently, and includes community context in answer generation.

---

## Critical Issues - ALL FIXED

### Issue #1: Docstrings Claim RRF, Implementation Uses Simple Boost

**Severity:** Critical
**Status:** FIXED (commit 6e8a353)

**The Problem:**
Code used simple boost-then-merge instead of proper RRF.

**The Fix:**
`hybrid_graph_retrieval()` now properly uses RRF:

```python
# query.py:698-704
rrf_result = reciprocal_rank_fusion(
    result_lists=[vector_search_results, graph_ranked_results],
    query_types=["vector", "graph"],
    k=GRAPHRAG_RRF_K,
    top_k=top_k,
)
```

Graph results are ranked by `path_length` (shorter = higher rank) via `_build_graph_ranked_list()`.

---

### Issue #2: Graph-Only Chunks Never Fetched

**Severity:** Critical
**Status:** FIXED (commit 6e8a353)

**The Problem:**
Graph traversal found chunk IDs but they were never fetched from Weaviate.

**The Fix:**
`fetch_chunks_by_ids()` now fetches ALL graph-discovered chunks:

```python
# query.py:689
all_graph_chunks = fetch_chunks_by_ids(graph_chunk_ids, collection_name)
```

Uses batch filtering (`ContainsAny`) for efficient retrieval.

---

### Issue #3: Entity Lookup Normalization Mismatch

**Severity:** Critical
**Status:** FIXED (commit 6e8a353)

**The Problem:**
Python normalization (Unicode, stopwords, punctuation) didn't match Cypher's `toLower(trim())`.

**The Fix:**
Query functions now pre-normalize in Python before Neo4j lookup:

```python
# neo4j_client.py:424-425
normalized_name = GraphEntity(name=entity_name, entity_type="").normalized_name()
# Then use $normalized_name in Cypher query
```

Both `find_entity_neighbors()` and `find_entities_by_names()` use this pattern.

---

### Issue #6: Community Context Not Reaching Generation

**Severity:** Medium
**Status:** FIXED (commit 81a5ce9)

**The Problem:**
`format_graph_context_for_generation()` existed but was never called. Community summaries were retrieved but not included in the LLM generation prompt.

**The Fix:**
Added `graph_context` parameter to `generate_answer()` and wired it up in `app.py`:

```python
# answer_generator.py - new parameter
def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    graph_context: Optional[str] = None,  # NEW
) -> GeneratedAnswer:

# app.py - wiring
graph_context = None
if graph_meta and not graph_meta.get("error"):
    graph_context = format_graph_context_for_generation(graph_meta)

answer = generate_answer(
    query=query,
    chunks=st.session_state.search_results,
    model=GENERATION_MODEL,
    graph_context=graph_context,
)
```

The generation prompt now includes community summaries as "Background" context.

---

## Medium Priority Issues - Documented Deviations

### Issue #4: No Map-Reduce for Global Search

**Severity:** Medium
**Status:** Documented deviation - NOT PLANNED

**Original Paper Design:**
```
Global Query → MAP: Each community generates partial answer → REDUCE: Aggregate
```

**This Implementation:**
Uses embedding similarity to find top-K community summaries, which are included as background context in generation.

**Justification:** Map-reduce would require N+1 additional LLM calls (one per community + reduce step), significantly increasing cost and latency. The current approach captures the essence of global queries at lower cost.

---

### Issue #5: Single Community Level Only

**Severity:** Medium
**Status:** Documented deviation - NOT PLANNED

**The Problem:**
Leiden produces hierarchical communities but only level 0 is used.

**Justification:** Hard to measure benefit of multiple levels, and query-time level selection adds complexity. Level 0 (finest granularity) provides good results for most queries.

---

## Dead Code Removed

### `enrich_results_with_graph()` - REMOVED (commit 76848fe)

This function was part of the old boost-based merge strategy. It was:
- Never called anywhere in the codebase
- Superseded by direct RRF merge in `hybrid_graph_retrieval()`

Removed from `query.py` and `__init__.py`.

---

## Summary Table

| # | Issue | Severity | Status | Commit |
|---|-------|----------|--------|--------|
| 1 | RRF not implemented | Critical | **FIXED** | 6e8a353 |
| 2 | Graph-only chunks not fetched | Critical | **FIXED** | 6e8a353 |
| 3 | Normalization mismatch | Critical | **FIXED** | 6e8a353 |
| 4 | No Map-Reduce global search | Medium | Documented deviation | - |
| 5 | Single community level | Medium | Documented deviation | - |
| 6 | Community context not in generation | Medium | **FIXED** | 81a5ce9 |

---

## Verification Tests

All fixes verified with automated tests:

| Test | Result |
|------|--------|
| Entity normalization consistency | PASS |
| RRF merging (shared chunks boosted) | PASS |
| LLM entity extraction | PASS |
| Graph ranking by path_length | PASS |
| Community embeddings present | PASS |
| generate_answer with graph_context | PASS |
| Backward compatibility (no context) | PASS |

---

## Current Implementation Flow

```
Query: "How does dopamine affect motivation?"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. ENTITY EXTRACTION (LLM)                                  │
│    Extracted: ["dopamine", "motivation"]                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. NEO4J LOOKUP (normalized)                                │
│    "dopamine" → normalized → matched in graph               │
│    Traverse 2 hops → find related chunk IDs                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PARALLEL RETRIEVAL                                       │
│    - Vector search (Weaviate) → top-k chunks                │
│    - Fetch ALL graph chunks (ContainsAny filter)            │
│    - Community retrieval (embedding similarity)             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. RRF MERGE                                                │
│    - Graph chunks ranked by path_length                     │
│    - RRF score = 1/(k+rank_vector) + 1/(k+rank_graph)       │
│    - Chunks in BOTH lists get boosted                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ANSWER GENERATION                                        │
│    Prompt includes:                                         │
│    - Background: community summaries + entity relationships │
│    - Retrieved Passages: RRF-merged chunks                  │
│    - Question: original query                               │
└─────────────────────────────────────────────────────────────┘
```

---

*Analysis Date: 2026-01-07*
*Fixes Completed: 2026-01-08*
