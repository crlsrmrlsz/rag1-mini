# GraphRAG Improvements: Final Implementation Plan

**Date:** 2025-12-28 (updated 2025-12-29)
**Status:** IMPLEMENTING v2
**References:**
- `graphrag-research.md` - Background analysis
- Implementation plan: `.claude/plans/async-gliding-allen.md`
- Crash-proof design: `docs/preprocessing/graphrag.md#crash-proof-design-v2`

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Enhanced Entity Resolution | COMPLETE |
| Phase 2 | Community Embedding Retrieval | COMPLETE |
| Phase 3 | Verification and Documentation | COMPLETE |
| **Phase 4** | **Crash-Proof Design + Weaviate Storage** | **IN PROGRESS** |

**Tests:** 30 tests passing (`pytest tests/ -v`)

---

## Phase 4: Crash-Proof Design (v2)

**Added:** 2025-12-29

### Problem Statement

Stage 6b takes ~10 hours and costs ~$10. Three critical issues:

1. **Leiden non-determinism** - Each run produces different community IDs
2. **383MB JSON file** - Embeddings stored inline (81% of file size)
3. **Resume broken after Neo4j reset** - Community IDs mismatch

### Solution

1. **Deterministic Leiden** - Add `randomSeed=42` + `concurrency=1`
2. **Weaviate storage** - Store community embeddings in Weaviate, not JSON
3. **Checkpoint file** - Small JSON with Leiden assignments

### Code Changes

| File | Change |
|------|--------|
| `config.py` | Add `LEIDEN_SEED = 42` |
| `community.py:run_leiden()` | Add `randomSeed`, `concurrency=1` |
| `community.py` | Add `save_leiden_checkpoint()`, `load_leiden_checkpoint()` |
| `weaviate_client.py` | Add `create_community_collection()`, `upload_community()`, `get_existing_community_ids()` |
| `query.py` | Change `retrieve_community_context()` to query Weaviate |
| `run_stage_6b_neo4j.py` | Use new checkpoint/Weaviate workflow |

### Storage Comparison

| Storage | Before | After |
|---------|--------|-------|
| communities.json | 383 MB | 0 |
| leiden_checkpoint.json | N/A | ~2 MB |
| Weaviate collection | N/A | ~10 MB |
| **Total** | **383 MB** | **~12 MB** |

### Crash Recovery

| Crash Point | Recovery |
|-------------|----------|
| During Upload | Re-run (MERGE idempotent) |
| During Leiden | Re-run (deterministic with seed) |
| During Summarization | `--resume` (skips existing in Weaviate) |
| Neo4j deleted | Re-upload + Leiden (same seed = same IDs), resume |

---

---

## Summary

Two focused improvements with measurable impact:
1. **Enhanced Entity Resolution** - Fix duplicate entities via better normalization
2. **Community Embedding Retrieval** - Replace keyword matching with embedding similarity

**User Choices:**
- Add embeddings for community retrieval
- Quality over speed (proper tests, documentation)

**Cost Optimization:** No re-extraction needed! The normalization happens at Neo4j upload time (Stage 6b), not during LLM extraction. We can apply improved normalization to existing `extraction_results.json` by simply re-running Stage 6b with updated code.

---

## Critical Analysis of Proposed Improvements

### P1: Enhanced Entity Resolution - RECOMMENDED

**Current Problem:**
```python
# src/graph/schemas.py:66-68
def normalized_name(self) -> str:
    return self.name.strip().lower()
```
- "The Dopamine" → "the dopamine"
- "dopamine" → "dopamine"
- Result: **Two different nodes** when they should merge

**Measurability:** HIGH
- Count unique entities before/after re-extraction
- Count graph edges before/after
- Specific test cases: "Marcus Aurelius" vs "Aurelius", "The prefrontal cortex" vs "prefrontal cortex"

**Effort:** LOW (~1-2 hours)
- Just update one function + add unit tests

**Verdict:** DO THIS - Low effort, high measurable impact, teaches text normalization

---

### P2: Map-Reduce Global Search - SIMPLIFY

**Current Problem:**
```python
# src/graph/query.py:284-296
query_words = set(query.lower().split())
summary_words = set(community.summary.lower().split())
overlap = len(query_words & summary_words)  # Simple word overlap
```
- Returns only top 3 communities by keyword match
- No semantic understanding
- No partial answer aggregation

**Full Map-Reduce (from paper):**
1. Shuffle all communities into 8k-token chunks
2. Generate partial answers in parallel with 0-100 scores
3. Filter zero-score answers
4. Sort by score, synthesize final answer

**Why Full Version is Overkill:**
- Multiple parallel LLM calls = higher cost
- Scoring system adds complexity
- For learning project, the concept matters more than production optimization

**Simplified Alternative:**
Instead of map-reduce, add **embedding similarity** for community retrieval:
- Community.embedding field already exists (never populated)
- Reuse existing `embedder.py`
- Replace keyword matching with cosine similarity

**Measurability:** HIGH
- Compare retrieval quality: keyword vs embedding
- Count relevant communities retrieved

**Effort:** MEDIUM (~2-3 hours)

**Verdict:** DO SIMPLIFIED VERSION - Embedding retrieval instead of map-reduce

---

### P3: Community Hierarchy - SKIP

**Current Problem:**
```python
# src/graph/community.py:445
level=0,  # Single level for now
```
- Leiden returns hierarchy (`intermediateCommunityIds`) but it's discarded
- All communities stored at level=0

**Why Skip:**
- Small corpus (few books) = shallow hierarchy (1-2 levels)
- "Right level selection" is hard to measure
- Adds significant complexity for unclear benefit

**Measurability:** LOW
- Hard to prove "level X was better than level Y"

**Effort:** HIGH (~4-5 hours)

**Verdict:** SKIP - Complexity not justified for learning project

---

### P4: Community Embeddings - ALREADY COVERED BY P2 SIMPLIFICATION

If we do the simplified P2 (embedding similarity), this is automatically included.

---

### P5: Self-Reflection Loop - SKIP

**Original Paper Approach:**
- After extraction, ask LLM: "Did you miss any entities?"
- If yes, extract more; repeat up to 3 times
- Claims ~20% more entities

**Why Skip:**
- 3x more LLM calls per chunk = 3x extraction cost
- Marginal benefit for learning project
- Existing extraction already gets the main entities

**Measurability:** MEDIUM (entity count increase)

**Effort:** MEDIUM (~2-3 hours)

**Verdict:** SKIP - Cost increase not justified for 20% improvement

---

## Final Implementation Plan

### Phase 1: Enhanced Entity Resolution (NO RE-EXTRACTION NEEDED)

**Goal:** Reduce duplicate entities by improving normalization at Neo4j upload time.

**Key Discovery:** Entity normalization happens in TWO places:
1. **Python:** `GraphEntity.normalized_name()` in `schemas.py` (currently: `strip().lower()`)
2. **Cypher:** `toLower(trim(entity.name))` in `neo4j_client.py` upload queries

The Cypher version is used during upload, so we need to:
1. Improve Python `normalized_name()`
2. Modify `upload_entities()` to pass Python-normalized name to Cypher
3. Re-run Stage 6b (fast, no LLM cost!)

#### Step 1.1: Update normalized_name() in schemas.py

**File:** `src/graph/schemas.py`

```python
import unicodedata
import re

# Add constant at module level
EDGE_STOPWORDS = frozenset({'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'and'})

def normalized_name(self) -> str:
    """Normalize entity name for deduplication."""
    name = self.name.strip()
    name = unicodedata.normalize('NFKC', name)
    name = name.lower()

    words = name.split()
    while words and words[0] in EDGE_STOPWORDS:
        words.pop(0)
    while words and words[-1] in EDGE_STOPWORDS:
        words.pop()

    name = ' '.join(words)
    name = re.sub(r'[^\w\s]', '', name)
    return ' '.join(name.split())
```

#### Step 1.2: Modify upload_entities() to use Python normalization

**File:** `src/graph/neo4j_client.py`

Current (line 224):
```python
MERGE (e:Entity {normalized_name: toLower(trim(entity.name))})
```

Change to pass pre-computed normalized name:
```python
def upload_entities(driver, entities, batch_size=100):
    # Pre-compute normalized names in Python
    from src.graph.schemas import GraphEntity

    for entity in entities:
        ge = GraphEntity(name=entity["name"], entity_type=entity.get("entity_type", ""))
        entity["normalized_name"] = ge.normalized_name()

    query = """
    UNWIND $entities AS entity
    MERGE (e:Entity {normalized_name: entity.normalized_name})
    ...
    """
```

Also update `upload_relationships()` lines 281-282 similarly.

#### Step 1.3: Add unit tests

**File:** `tests/test_entity_resolution.py` (new)

```python
import pytest
from src.graph.schemas import GraphEntity

@pytest.mark.parametrize("input_name,expected", [
    ("The Dopamine", "dopamine"),
    ("dopamine", "dopamine"),
    ("café", "cafe"),
    ("Marcus Aurelius' Meditations", "marcus aurelius meditations"),
    ("The Art of War", "art war"),
    ("Prefrontal Cortex, The", "prefrontal cortex"),
    ("the", ""),
])
def test_normalized_name(input_name, expected):
    entity = GraphEntity(name=input_name, entity_type="TEST")
    assert entity.normalized_name() == expected
```

#### Step 1.4: Capture baseline and re-upload

```bash
# Count entities in Neo4j BEFORE
docker exec neo4j_rag cypher-shell -u neo4j -p password \
  "MATCH (e:Entity) RETURN count(DISTINCT e.normalized_name) as unique_entities"

# Clear and re-upload (no re-extraction!)
python -m src.stages.run_stage_6b_neo4j

# Count entities AFTER
docker exec neo4j_rag cypher-shell -u neo4j -p password \
  "MATCH (e:Entity) RETURN count(DISTINCT e.normalized_name) as unique_entities"
```

**Expected:** 10-20% reduction in unique entity count due to better merging.

---

### Phase 2: Community Embedding Retrieval

**Goal:** Replace keyword matching with embedding similarity for community retrieval.

#### Step 2.1: Add embedding generation to community summarization

**File:** `src/graph/community.py`

After generating summary, call embedder:
```python
from src.rag_pipeline.embedding.embedder import embed_texts

def summarize_community(...) -> Tuple[str, List[float]]:
    """Generate summary AND embedding for a community."""
    # ... existing summary generation ...

    # Generate embedding for the summary
    embeddings = embed_texts([summary])
    embedding = embeddings[0] if embeddings else None

    return summary, embedding
```

Update `detect_and_summarize_communities()` to store embedding in Community object.

#### Step 2.2: Update save/load to handle embeddings

**File:** `src/graph/community.py`

Update `save_communities()`:
```python
def to_dict(self) -> Dict[str, Any]:
    return {
        # ... existing fields ...
        "embedding": self.embedding,  # Add embedding
    }
```

Update `load_communities()`:
```python
community = Community(
    # ... existing fields ...
    embedding=c_data.get("embedding"),  # Load embedding
)
```

#### Step 2.3: Replace keyword matching with embedding similarity

**File:** `src/graph/query.py`

```python
import numpy as np
from src.rag_pipeline.embedding.embedder import embed_texts

def retrieve_community_context(
    query: str,
    communities: Optional[List[Community]] = None,
    top_k: int = GRAPHRAG_TOP_COMMUNITIES,
) -> List[Dict[str, Any]]:
    """Retrieve relevant community summaries using embedding similarity."""
    if communities is None:
        communities = load_communities()

    if not communities:
        return []

    # Generate query embedding
    query_embedding = embed_texts([query])[0]

    # Score by cosine similarity
    scored = []
    for community in communities:
        if community.embedding:
            similarity = cosine_similarity(query_embedding, community.embedding)
            scored.append((similarity, community))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top-k
    return [
        {
            "community_id": c.community_id,
            "summary": c.summary,
            "member_count": c.member_count,
            "score": score,
        }
        for score, c in scored[:top_k]
    ]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
```

#### Step 2.4: Add tests for embedding retrieval

**File:** `tests/test_community_retrieval.py` (new)

- Test embedding generation for summaries
- Test similarity scoring
- Compare with previous keyword approach on sample queries

#### Step 2.5: Regenerate communities with embeddings

```bash
# Re-run with embedding generation
python -m src.stages.run_stage_6b_neo4j
```

---

### Phase 3: Verification and Documentation

#### Step 3.1: Run evaluation to compare

```bash
# Run evaluation with graphrag strategy
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag
```

#### Step 3.2: Update memory-bank

Add implementation notes to `memory-bank/graphrag-research.md`:
- Entity resolution improvements
- Community embedding retrieval
- Before/after metrics

---

## Files to Modify

| File | Change |
|------|--------|
| `src/graph/schemas.py` | Enhanced `normalized_name()` with Unicode/stopwords |
| `src/graph/neo4j_client.py` | Pass Python-normalized name to Cypher MERGE |
| `src/graph/community.py` | Add embedding generation during summarization |
| `src/graph/query.py` | Replace keyword matching with cosine similarity |
| `tests/test_entity_resolution.py` | New: unit tests for normalization |
| `tests/test_community_retrieval.py` | New: unit tests for embedding retrieval |
| `memory-bank/graphrag-research.md` | Update with implementation notes |

## Execution Order

1. **Phase 1.1-1.3:** Code changes + tests for entity resolution
2. **Phase 1.4:** Capture baseline, re-run Stage 6b, measure improvement
3. **Phase 2.1-2.3:** Code changes for embedding retrieval
4. **Phase 2.4-2.5:** Tests + regenerate communities with embeddings
5. **Phase 3:** Verification and documentation

---

## What We're NOT Doing

- **Full map-reduce** - Too complex for learning project; embedding retrieval captures the essence
- **Community hierarchy** - Hard to measure with small corpus; value unclear
- **Self-reflection loop** - 3x cost increase for ~20% more entities; not justified

---

## Expected Measurements

### Entity Resolution (P1)
- **Before:** ~X unique entities (with duplicates like "dopamine", "the dopamine", "Dopamine")
- **After:** ~Y unique entities (merged duplicates)
- **Target:** 10-20% reduction in entity count

### Community Retrieval (P2)
- **Before:** Keyword matching (word overlap score)
- **After:** Embedding similarity (cosine similarity)
- **Test:** "What are the main themes?" should retrieve thematically relevant communities, not just keyword matches

---

## Cost Summary

| Phase | LLM Calls | Time Est. |
|-------|-----------|-----------|
| P1: Entity Resolution | 0 (pure code) | ~2 hours |
| P1: Stage 6b Re-upload | 0 (no extraction) | ~5 min |
| P2: Embedding Generation | ~50 embed calls for community summaries | ~3 hours |
| P2: Tests | 0 | ~1 hour |
| P3: Verification | 1 eval run | ~30 min |

**Total LLM Cost:** Minimal (~50 embedding calls for community summaries)
**No re-extraction needed!**

---

*Last Updated: 2025-12-28*
