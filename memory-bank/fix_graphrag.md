# GraphRAG Implementation Analysis & Fix Plan

**Date:** 2026-01-07
**Status:** Analysis Complete - Fixes Pending
**Related:** [graphrag.md](graphrag.md), [graphrag-tutorial.md](graphrag-tutorial.md)

---

## Executive Summary

Deep analysis of GraphRAG implementation revealed several significant issues where **docstrings don't match actual implementation** and **key features are incomplete**. The most critical finding: documentation claims RRF (Reciprocal Rank Fusion) is used throughout, but the actual code uses a simple boost strategy.

---

## Critical Issues

### Issue #1: Docstrings Claim RRF, Implementation Uses Simple Boost

**Severity:** Critical
**Location:** `src/graph/query.py:495-565` and docstrings throughout

**The Problem:**

Multiple docstrings claim RRF (Reciprocal Rank Fusion) is used:
- `src/graph/__init__.py:25` - "Query → Graph traversal + Vector search → RRF merge → Answer"
- `src/graph/query.py:9` - "The hybrid approach uses RRF (Reciprocal Rank Fusion) to merge"
- `src/graph/query.py:19` - "RRF from src/rag_pipeline/retrieval/rrf.py"
- `src/config.py:402` - "Hybrid graph + vector retrieval via RRF"

**Actual implementation** in `hybrid_graph_retrieval()`:

```python
# Line 549-555 (query.py)
# Sort: graph-boosted results first, then by original score
# This is a simple boost strategy; could use RRF for more sophistication  <-- ADMITS IT!
boosted = [r for r in enriched if r.get("graph_boost")]
non_boosted = [r for r in enriched if not r.get("graph_boost")]
# Combine: boosted first (preserving their order), then non-boosted
merged = boosted + non_boosted
```

This is **NOT RRF** - it's simple list concatenation with boost flags.

**What RRF would look like:**
```python
# RRF formula: score(d) = Σ(1/(k + rank(d, list_i))) for each result list
# Would compute combined score from both vector ranks AND graph traversal ranks
```

**Fix Options:**
1. Implement proper RRF using existing `src/rag_pipeline/retrieval/rrf.py`
2. OR update all docstrings to accurately describe "boost-then-merge" strategy

---

### Issue #2: Graph-Only Chunks Never Fetched

**Severity:** Critical
**Location:** `src/graph/query.py:489-492, 559`

**The Problem:**

Graph traversal identifies chunks that vector search missed, but they're **never actually retrieved**:

```python
# enrich_results_with_graph() returns:
def enrich_results_with_graph(...) -> tuple[list[dict], set[str]]:
    ...
    # Find graph-only chunks not in vector results
    graph_only = graph_set - vector_chunk_ids
    return vector_results, graph_only  # graph_only returned but...

# In hybrid_graph_retrieval():
enriched, graph_only_ids = enrich_results_with_graph(...)
# ...
logger.info(f"... {len(graph_only_ids)} graph-only (not fetched)")  # Just logged!
```

**Impact:** This defeats a key purpose of GraphRAG - finding related content that vector similarity alone wouldn't surface. If graph traversal finds 20 relevant chunks not in vector results, they're simply discarded.

**Fix:** After `enrich_results_with_graph()`, fetch the missing chunks from Weaviate by chunk_id and include them in final results (possibly with lower initial rank for RRF).

---

### Issue #3: Entity Lookup Normalization Mismatch

**Severity:** Critical
**Location:** `src/graph/neo4j_client.py:420-421` vs `src/graph/schemas.py:72-101`

**The Problem:**

**During Upload (Python-side normalization in `GraphEntity.normalized_name()`):**
```python
def normalized_name(self) -> str:
    name = self.name.strip()
    name = unicodedata.normalize('NFKC', name)  # Unicode normalization
    name = name.lower()

    # Remove leading/trailing stopwords
    words = name.split()
    while words and words[0] in EDGE_STOPWORDS:  # the, a, an, of, in, on, for, to, and
        words.pop(0)
    while words and words[-1] in EDGE_STOPWORDS:
        words.pop()

    name = ' '.join(words)
    name = re.sub(r'[^\w\s]', '', name)  # Strip punctuation
    return ' '.join(name.split())  # Normalize whitespace
```

**During Query (Cypher-side normalization in `find_entity_neighbors()`):**
```python
query = f"""
MATCH (start:Entity {{normalized_name: toLower(trim($entity_name))}})
...
"""
# Only applies: toLower() + trim() - NO stopwords, NO punctuation, NO Unicode!
```

**Example of mismatch:**
- Entity uploaded as "The Dopamine System" → normalized to `"dopamine system"`
- Query for "the dopamine system" → Cypher normalizes to `"the dopamine system"`
- **No match!**

**Fix:** Pre-normalize query entities in Python before Neo4j lookup:
```python
from src.graph.schemas import GraphEntity

def find_entity_neighbors(driver, entity_name, ...):
    # Normalize using same logic as upload
    normalized = GraphEntity(name=entity_name, entity_type="").normalized_name()
    query = f"""
    MATCH (start:Entity {{normalized_name: $normalized_name}})
    ...
    """
    result = driver.execute_query(query, normalized_name=normalized, ...)
```

---

## Medium Priority Issues

### Issue #4: No Map-Reduce for Global Search

**Severity:** Medium
**Location:** `src/graph/query.py:277-384`

**Original Paper Design:**
```
Global Query → MAP: Each community generates partial answer → REDUCE: Aggregate all partial answers
```

**Actual Implementation:**
- Uses embedding similarity to find top-K community summaries
- Summaries returned as context but no map-reduce LLM aggregation
- The documentation acknowledges this deviation (graphrag.md:227)

**Impact:** Less comprehensive answers for global queries like "What are the main themes across all documents?"

**Fix:** Implement optional map-reduce mode:
1. For each top-K community, generate a partial answer
2. Aggregate partial answers into final response
3. More expensive but more thorough for global queries

---

### Issue #5: Single Community Level Only

**Severity:** Medium
**Location:** `src/graph/community.py:622-624`

**The Problem:**

Leiden produces hierarchical communities (multiple levels), but only level 0 is used:

```python
community = Community(
    community_id=community_key,
    level=0,  # Single level for now  <-- HARDCODED
    ...
)
```

The `intermediate_ids` from Leiden (line 160) are captured but never utilized:
```python
node_communities.append({
    "node_id": record.nodeId,
    "community_id": record.communityId,
    "intermediate_ids": list(record.intermediateCommunityIds),  # Captured but unused
})
```

**Impact:** Loses hierarchical structure that could enable coarse-to-fine retrieval.

**Fix:** Store and use multiple levels, allow query-time level selection based on query scope.

---

### Issue #6: Community Context May Not Reach Generation

**Severity:** Medium
**Location:** `src/ui/services/search.py:251-277`, `src/graph/query.py:568-607`

**The Problem:**

`format_graph_context_for_generation()` exists to format community summaries for LLM context:

```python
def format_graph_context_for_generation(metadata: dict, max_chars: int = 2000) -> str:
    """Format graph metadata as additional context for answer generation."""
    lines = []
    if metadata.get("community_context"):
        lines.append("## Relevant Themes (from document corpus)")
        for comm in metadata["community_context"][:2]:
            lines.append(f"\n{comm['summary']}")
    ...
```

But in the search flow, `graph_metadata` is returned and it's unclear if community summaries are actually added to the generation prompt or just returned as metadata for logging.

**Fix:** Verify community summaries are included in generation context, or explicitly add them.

---

## Summary Table

| # | Issue | Severity | Status | Files Affected |
|---|-------|----------|--------|----------------|
| 1 | Docstrings claim RRF, uses simple boost | Critical | Pending | `query.py`, `__init__.py`, `config.py` |
| 2 | Graph-only chunks never fetched | Critical | Pending | `query.py` |
| 3 | Normalization mismatch Python vs Cypher | Critical | Pending | `neo4j_client.py`, `schemas.py` |
| 4 | No Map-Reduce for global search | Medium | Documented deviation | `query.py` |
| 5 | Single community level only | Medium | Pending | `community.py` |
| 6 | Community context usage unclear | Medium | Needs verification | `search.py` |

---

## Recommended Fix Order

1. **Issue #3 (Normalization)** - Quick fix, prevents silent failures
2. **Issue #2 (Graph-only chunks)** - Medium effort, significant impact
3. **Issue #1 (RRF vs Boost)** - Either implement RRF OR fix docstrings
4. **Issues #4-6** - Lower priority, can be deferred

---

## Code Locations Reference

```
src/graph/
├── __init__.py          # Line 25: RRF claim in docstring
├── query.py             # Lines 9, 19, 26: RRF claims; 495-565: actual boost logic
├── neo4j_client.py      # Lines 420-421: Cypher normalization
├── schemas.py           # Lines 72-101: Python normalization
├── community.py         # Lines 622-624: hardcoded level=0
└── extractor.py         # OK

src/ui/services/
└── search.py            # Lines 251-277: GraphRAG integration

src/config.py            # Line 402: RRF claim
```

---

*Analysis Date: 2026-01-07*
