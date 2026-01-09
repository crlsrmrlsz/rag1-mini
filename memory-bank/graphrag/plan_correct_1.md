# GraphRAG Improvement Plan: Hierarchy, PageRank, and Structured Relationships

**Date:** 2026-01-09
**Status:** Planning
**Priority:** Medium
**Estimated Scope:** 3 features, ~500 lines of code changes

---

## Executive Summary

This plan addresses three gaps between the current GraphRAG implementation and the original Microsoft Research paper (arXiv:2404.16130). These improvements will enable better global query handling, smarter entity prioritization, and structured relationship access at query time.

| Gap | Current State | Target State |
|-----|---------------|--------------|
| Hierarchical Communities | Level 0 only | Levels 0, 1, 2+ with summaries |
| PageRank Centrality | Schema exists, never computed | Computed and used for ranking |
| Relationships in JSON | Count only, description in summary | Structured list in JSON |

---

## Background: Original GraphRAG vs Current Implementation

### Original Paper Design

The Microsoft GraphRAG paper describes:

1. **Hierarchical Communities (C0, C1, C2...)**
   - Leiden algorithm produces multiple levels naturally
   - Level 0 = finest granularity (small, specific clusters)
   - Level 1+ = coarser (aggregated clusters for global queries)
   - Query routing: local queries use L0, global queries use L1/L2
   - Map-reduce: each community generates partial answer, then aggregated

2. **Entity Importance Ranking**
   - PageRank or similar centrality measure
   - Hub entities prioritized in context window
   - More informative than raw degree count

3. **Structured Knowledge**
   - Entities AND relationships stored as structured data
   - Direct access without graph database queries
   - Enables relationship-based filtering

### Current Implementation State

**Files involved:**
- `src/graph/community.py` - Leiden + summarization
- `src/graph/schemas.py` - Pydantic models
- `src/graph/query.py` - Query-time retrieval
- `src/config.py` - Configuration parameters

**Current behavior:**

1. **Hierarchy:** Leiden runs with `includeIntermediateCommunities=True` (line 149), captures `intermediateCommunityIds` (line 160-161), but **only level 0 is used** (hardcoded at line 624).

2. **PageRank:** Schema defines `pagerank: float = Field(default=0.0)` in `CommunityMember` (line 269), but **never computed**. Always 0.0.

3. **Relationships:** `get_community_relationships()` fetches from Neo4j (lines 365-388), used in `build_community_context()` for LLM prompt (lines 391-432), but **not stored in JSON**. Only `relationship_count` is saved.

---

## Gap 1: Hierarchical Communities

### Problem Statement

Only level 0 communities are summarized and stored. Global queries like "What are the main themes across all books?" cannot leverage coarser community levels that would provide broader thematic summaries.

### Current Code Analysis

```python
# community.py:149-161 - Leiden DOES produce hierarchy
result = gds.leiden.stream(
    graph,
    includeIntermediateCommunities=True,  # ← Hierarchy captured!
)

for record in result.itertuples():
    node_communities.append({
        "node_id": record.nodeId,
        "community_id": record.communityId,
        "intermediate_ids": list(record.intermediateCommunityIds),  # ← Stored but IGNORED
    })

# community.py:622-624 - Only level 0 used
community = Community(
    community_id=community_key,
    level=0,  # ← HARDCODED
    ...
)
```

### Implementation Plan

#### Step 1: Parse Hierarchy from Leiden Result

**File:** `src/graph/community.py`

```python
def parse_community_hierarchy(node_communities: list[dict]) -> dict[int, dict[int, set]]:
    """Parse Leiden result into level -> community_id -> node_ids mapping.

    Args:
        node_communities: List from run_leiden() with intermediate_ids.

    Returns:
        Dict: {level: {community_id: {node_id, ...}}}

    Example:
        >>> hierarchy = parse_community_hierarchy(leiden_result["node_communities"])
        >>> hierarchy[0]  # Level 0: {42: {1, 2, 3}, 43: {4, 5}, ...}
        >>> hierarchy[1]  # Level 1: {10: {1, 2, 3, 4, 5}, ...}
    """
    hierarchy = defaultdict(lambda: defaultdict(set))

    for nc in node_communities:
        node_id = nc["node_id"]
        # Level 0 is the final communityId
        hierarchy[0][nc["community_id"]].add(node_id)

        # Intermediate levels (index 0 = level 1, etc.)
        for level_idx, comm_id in enumerate(nc["intermediate_ids"]):
            hierarchy[level_idx + 1][comm_id].add(node_id)

    return dict(hierarchy)
```

#### Step 2: Add Level Parameter to Summarization

**File:** `src/graph/community.py`

Modify `detect_and_summarize_communities()`:

```python
def detect_and_summarize_communities(
    driver: Driver,
    gds: GraphDataScience,
    min_size: int = GRAPHRAG_MIN_COMMUNITY_SIZE,
    model: str = GRAPHRAG_SUMMARY_MODEL,
    resume: bool = False,
    skip_leiden: bool = False,
    max_level: int = 2,  # NEW: How many levels to summarize
) -> dict[int, list[Community]]:
    """Returns dict of {level: [Community, ...]}"""

    # ... existing Leiden code ...

    hierarchy = parse_community_hierarchy(leiden_result["node_communities"])

    all_communities = {}
    for level in range(min(max_level + 1, len(hierarchy))):
        level_communities = []
        for community_id, node_ids in hierarchy[level].items():
            if len(node_ids) < min_size:
                continue
            # Summarize this community at this level
            members = get_community_members_by_node_ids(driver, node_ids)
            relationships = get_community_relationships_by_node_ids(driver, node_ids)
            summary, embedding = summarize_community(members, relationships, model)

            community = Community(
                community_id=f"community_L{level}_{community_id}",
                level=level,
                members=members,
                # ...
            )
            level_communities.append(community)

        all_communities[level] = level_communities

    return all_communities
```

#### Step 3: New Helper Functions

**File:** `src/graph/community.py`

```python
def get_community_members_by_node_ids(
    driver: Driver,
    node_ids: set[int],
) -> list[CommunityMember]:
    """Get members by Neo4j internal node IDs (for hierarchy levels)."""
    query = """
    MATCH (e:Entity)
    WHERE id(e) IN $node_ids
    OPTIONAL MATCH (e)-[r:RELATED_TO]-()
    WITH e, count(r) as degree
    RETURN
        e.name as entity_name,
        e.entity_type as entity_type,
        e.description as description,
        e.pagerank as pagerank,
        degree
    ORDER BY pagerank DESC, degree DESC
    """
    # ... implementation ...


def get_community_relationships_by_node_ids(
    driver: Driver,
    node_ids: set[int],
) -> list[dict]:
    """Get relationships between nodes in a set (for hierarchy levels)."""
    query = """
    MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
    WHERE id(source) IN $node_ids AND id(target) IN $node_ids
    RETURN
        source.name as source,
        target.name as target,
        r.type as relationship_type,
        r.description as description
    """
    # ... implementation ...
```

#### Step 4: Update Storage

**File:** `src/graph/community.py`

```python
def save_communities(
    communities: dict[int, list[Community]],  # Changed from list to dict
    output_name: str = "communities.json",
) -> Path:
    """Save hierarchical community data."""
    data = {
        "levels": {
            str(level): [c.to_dict() for c in comms]
            for level, comms in communities.items()
        },
        "level_counts": {str(l): len(c) for l, c in communities.items()},
        "total_communities": sum(len(c) for c in communities.values()),
    }
    # ... save to JSON ...
```

#### Step 5: Configuration

**File:** `src/config.py`

```python
# Leiden hierarchy
GRAPHRAG_MAX_COMMUNITY_LEVEL = 2  # Summarize levels 0, 1, 2
GRAPHRAG_LEVEL_MIN_SIZES = {
    0: 3,   # L0: min 3 members (fine-grained)
    1: 5,   # L1: min 5 members (medium)
    2: 10,  # L2: min 10 members (coarse)
}
```

---

## Gap 2: PageRank Centrality

### Problem Statement

Entity importance within communities is only measured by `degree` (connection count). PageRank would identify hub entities that connect to other important entities, providing better prioritization for context windows.

### Current Code Analysis

```python
# schemas.py:269 - Field exists but never populated
pagerank: float = Field(default=0.0, description="PageRank centrality score")

# community.py:339-347 - Only degree is computed
query = """
MATCH (e:Entity {community_id: $community_id})
OPTIONAL MATCH (e)-[r:RELATED_TO]-()
WITH e, count(r) as degree
RETURN ..., degree
# NO pagerank!
"""
```

### Implementation Plan

#### Step 1: Run PageRank After Leiden

**File:** `src/graph/community.py`

```python
def run_pagerank(
    gds: GraphDataScience,
    graph: Any,
    damping_factor: float = 0.85,
    max_iterations: int = 20,
) -> dict[int, float]:
    """Run PageRank on the projected graph.

    Args:
        gds: GraphDataScience client.
        graph: GDS graph projection.
        damping_factor: PageRank damping (default 0.85).
        max_iterations: Max iterations (default 20).

    Returns:
        Dict mapping node_id -> pagerank score.
    """
    result = gds.pageRank.stream(
        graph,
        dampingFactor=damping_factor,
        maxIterations=max_iterations,
    )

    scores = {}
    for record in result.itertuples():
        scores[record.nodeId] = record.score

    logger.info(f"Computed PageRank for {len(scores)} nodes")
    return scores
```

#### Step 2: Write PageRank to Neo4j

**File:** `src/graph/community.py`

```python
def write_pagerank_to_neo4j(
    driver: Driver,
    pagerank_scores: dict[int, float],
) -> int:
    """Write PageRank scores to Entity nodes.

    Args:
        driver: Neo4j driver.
        pagerank_scores: Dict from run_pagerank().

    Returns:
        Number of nodes updated.
    """
    assignments = [
        {"node_id": node_id, "score": score}
        for node_id, score in pagerank_scores.items()
    ]

    query = """
    UNWIND $assignments AS assignment
    MATCH (e:Entity)
    WHERE id(e) = assignment.node_id
    SET e.pagerank = assignment.score
    RETURN count(e) as count
    """

    result = driver.execute_query(query, assignments=assignments)
    count = result.records[0]["count"]
    logger.info(f"Updated {count} nodes with PageRank scores")
    return count
```

#### Step 3: Update get_community_members Query

**File:** `src/graph/community.py`

```python
def get_community_members(
    driver: Driver,
    community_id: int,
) -> list[CommunityMember]:
    query = """
    MATCH (e:Entity {community_id: $community_id})
    OPTIONAL MATCH (e)-[r:RELATED_TO]-()
    WITH e, count(r) as degree
    RETURN
        e.name as entity_name,
        e.entity_type as entity_type,
        e.description as description,
        degree,
        coalesce(e.pagerank, 0.0) as pagerank  # ← ADD THIS
    ORDER BY pagerank DESC, degree DESC  # ← Sort by PageRank first
    """

    result = driver.execute_query(query, community_id=community_id)

    members = []
    for record in result.records:
        members.append(CommunityMember(
            entity_name=record["entity_name"],
            entity_type=record["entity_type"] or "UNKNOWN",
            description=record["description"] or "",
            degree=record["degree"],
            pagerank=record["pagerank"],  # ← ADD THIS
        ))

    return members
```

#### Step 4: Integrate into Pipeline

**File:** `src/graph/community.py`

In `detect_and_summarize_communities()`, after Leiden and before summarization:

```python
# Step 2: Run Leiden
leiden_result = run_leiden(gds, graph)

# Step 2.5: Run PageRank (NEW)
pagerank_scores = run_pagerank(gds, graph)
write_pagerank_to_neo4j(driver, pagerank_scores)

# Step 3: Save checkpoint
save_leiden_checkpoint(leiden_result)
# ... rest of pipeline ...
```

---

## Gap 3: Relationships in JSON

### Problem Statement

Relationships are fetched for LLM summarization but not stored in the JSON output. Only `relationship_count` is saved. At query time, we cannot access structured relationship data without querying Neo4j.

### Current Code Analysis

```python
# community.py:365-388 - Relationships ARE fetched
def get_community_relationships(driver, community_id):
    query = """
    MATCH (source:Entity {community_id: $community_id})
          -[r:RELATED_TO]->
          (target:Entity {community_id: $community_id})
    RETURN source.name, target.name, r.type, r.description
    """
    # Returns list of dicts

# community.py:391-432 - Used in LLM prompt
def build_community_context(members, relationships, max_tokens):
    # Formats relationships for LLM
    for rel in relationships:
        lines.append(f"- {rel['source']} --[{rel['relationship_type']}]--> {rel['target']}")

# schemas.py:314-324 - NOT stored in to_dict()
def to_dict(self) -> dict[str, Any]:
    return {
        "community_id": self.community_id,
        "level": self.level,
        "member_count": self.member_count,
        "relationship_count": self.relationship_count,  # ← Only count!
        "summary": self.summary,
        "embedding": self.embedding,
        "members": [m.model_dump() for m in self.members],
        # NO "relationships" field!
    }
```

### Implementation Plan

#### Step 1: Add Relationship Schema

**File:** `src/graph/schemas.py`

```python
class CommunityRelationship(BaseModel):
    """Relationship within a community.

    Stores structured relationship data for direct access at query time
    without requiring Neo4j queries.

    Attributes:
        source: Source entity name.
        target: Target entity name.
        relationship_type: Type of relationship (e.g., "CAUSES", "MODULATES").
        description: Optional description of the relationship.
        weight: Relationship strength (0.0-1.0).
    """

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relationship_type: str = Field(..., description="Relationship type")
    description: str = Field(default="", description="Relationship description")
    weight: float = Field(default=1.0, description="Relationship strength")
```

#### Step 2: Update Community Schema

**File:** `src/graph/schemas.py`

```python
class Community(BaseModel):
    # ... existing fields ...

    relationships: list[CommunityRelationship] = Field(
        default_factory=list,
        description="Relationships within this community",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_id": self.community_id,
            "level": self.level,
            "member_count": self.member_count,
            "relationship_count": self.relationship_count,
            "summary": self.summary,
            "embedding": self.embedding,
            "members": [m.model_dump() for m in self.members],
            "relationships": [r.model_dump() for r in self.relationships],  # ← ADD
        }
```

#### Step 3: Update get_community_relationships Return Type

**File:** `src/graph/community.py`

```python
def get_community_relationships(
    driver: Driver,
    community_id: int,
) -> list[CommunityRelationship]:  # ← Changed return type
    """Get relationships within a community as structured objects."""
    query = """
    MATCH (source:Entity {community_id: $community_id})
          -[r:RELATED_TO]->
          (target:Entity {community_id: $community_id})
    RETURN
        source.name as source,
        target.name as target,
        r.type as relationship_type,
        r.description as description,
        coalesce(r.weight, 1.0) as weight
    """

    result = driver.execute_query(query, community_id=community_id)

    return [
        CommunityRelationship(
            source=r["source"],
            target=r["target"],
            relationship_type=r["relationship_type"] or "RELATED_TO",
            description=r["description"] or "",
            weight=r["weight"],
        )
        for r in result.records
    ]
```

#### Step 4: Pass Relationships to Community Object

**File:** `src/graph/community.py`

In `detect_and_summarize_communities()`:

```python
# Get relationships (now returns CommunityRelationship objects)
relationships = get_community_relationships(driver, community_id)

# ... summarization ...

community = Community(
    community_id=community_key,
    level=0,
    members=members,
    member_count=len(members),
    relationship_count=len(relationships),
    summary=summary,
    embedding=embedding,
    relationships=relationships,  # ← ADD THIS
)
```

#### Step 5: Update load_communities

**File:** `src/graph/community.py`

```python
def load_communities(input_name: str = "communities.json") -> list[Community]:
    # ... existing code ...

    for c_data in data["communities"]:
        members = [CommunityMember(**m) for m in c_data.get("members", [])]
        relationships = [
            CommunityRelationship(**r)
            for r in c_data.get("relationships", [])
        ]  # ← ADD THIS

        community = Community(
            # ... existing fields ...
            relationships=relationships,  # ← ADD THIS
        )
```

---

## Query Phase Changes

### Current Query Flow

```
Query → Entity Extraction → Neo4j Traversal → Vector Search → RRF Merge → Answer
                                    ↓
                           Community Retrieval (L0 only, no PageRank, no structured rels)
```

### Enhanced Query Flow

```
Query → Classify Scope (local/global) → Select Community Level
                                              ↓
                    ┌─────────────────────────┴─────────────────────────┐
                    ↓                                                   ↓
            Local Query (L0)                                    Global Query (L2)
                    ↓                                                   ↓
        Entity Extraction + Traversal                          Community Summaries
                    ↓                                                   ↓
              Vector Search                                     Map: Per-community answers
                    ↓                                                   ↓
               RRF Merge                                        Reduce: Aggregate
                    ↓                                                   ↓
    Community Context (PageRank-sorted members,              Final Answer (themes)
     structured relationships)
                    ↓
              Answer Gen
```

### Implementation: Query-Time Changes

**File:** `src/graph/query.py`

#### 1. Query Scope Classification

```python
def classify_query_scope(query: str) -> str:
    """Classify query as 'local' or 'global'.

    Local: Specific questions about entities/facts
    Global: Broad questions about themes/patterns

    Args:
        query: User query string.

    Returns:
        "local" or "global"
    """
    global_indicators = [
        "main themes", "overall", "across all", "summarize",
        "what are the", "general", "broad", "corpus-wide",
        "common patterns", "key takeaways",
    ]

    query_lower = query.lower()
    for indicator in global_indicators:
        if indicator in query_lower:
            return "global"

    return "local"
```

#### 2. Level-Aware Community Retrieval

```python
def retrieve_community_context(
    query: str,
    level: int = 0,  # NEW: Community level to query
    top_k: int = GRAPHRAG_TOP_COMMUNITIES,
) -> list[dict[str, Any]]:
    """Retrieve communities at specified level.

    Args:
        query: User query.
        level: Hierarchy level (0=fine, 1=medium, 2=coarse).
        top_k: Number of communities to return.

    Returns:
        List of community dicts with PageRank-sorted members
        and structured relationships.
    """
    collection_name = get_community_collection_name(level=level)

    # ... vector similarity search ...

    results = []
    for community in top_communities:
        # Members already sorted by PageRank from storage
        top_members = community.members[:10]  # High PageRank first

        # Structured relationships available directly
        top_relationships = community.relationships[:15]

        results.append({
            "community_id": community.community_id,
            "level": community.level,
            "summary": community.summary,
            "members": [m.model_dump() for m in top_members],
            "relationships": [r.model_dump() for r in top_relationships],
            "score": similarity_score,
        })

    return results
```

#### 3. Enhanced format_graph_context_for_generation

```python
def format_graph_context_for_generation(
    metadata: dict[str, Any],
    max_chars: int = 2000,
) -> str:
    """Format graph context with structured relationships."""
    lines = []

    # Community summaries
    if metadata.get("community_context"):
        lines.append("## Relevant Themes")
        for comm in metadata["community_context"][:2]:
            lines.append(f"\n### {comm.get('community_id', 'Community')}")
            lines.append(comm['summary'])

            # Add top relationships (structured data!)
            if comm.get("relationships"):
                lines.append("\nKey Relationships:")
                for rel in comm["relationships"][:5]:
                    lines.append(
                        f"  - {rel['source']} --[{rel['relationship_type']}]--> {rel['target']}"
                    )

            # Add top members by PageRank
            if comm.get("members"):
                lines.append("\nKey Entities (by importance):")
                for member in comm["members"][:5]:
                    pr = member.get("pagerank", 0)
                    lines.append(f"  - {member['entity_name']} (PageRank: {pr:.3f})")

    # ... rest of function ...
```

#### 4. Global Query Handler (Map-Reduce)

```python
def handle_global_query(
    query: str,
    model: str = GENERATION_MODEL,
) -> str:
    """Handle global queries using map-reduce over L2 communities.

    Map: Generate partial answer from each community
    Reduce: Aggregate into final answer
    """
    # Get coarse communities (L2)
    communities = retrieve_community_context(query, level=2, top_k=10)

    # MAP: Generate partial answers
    partial_answers = []
    for community in communities:
        context = f"""
        Community Theme: {community['summary']}

        Key Entities: {', '.join(m['entity_name'] for m in community['members'][:5])}

        Key Relationships:
        {chr(10).join(f"- {r['source']} {r['relationship_type']} {r['target']}"
                      for r in community['relationships'][:5])}
        """

        partial = call_chat_completion(
            messages=[{
                "role": "user",
                "content": f"Based on this community context, briefly answer: {query}\n\n{context}"
            }],
            model=model,
            max_tokens=200,
        )
        partial_answers.append(partial)

    # REDUCE: Aggregate
    reduce_prompt = f"""
    Question: {query}

    Partial answers from different thematic communities:
    {chr(10).join(f'{i+1}. {ans}' for i, ans in enumerate(partial_answers))}

    Synthesize these into a comprehensive final answer.
    """

    final_answer = call_chat_completion(
        messages=[{"role": "user", "content": reduce_prompt}],
        model=model,
        max_tokens=500,
    )

    return final_answer
```

#### 5. Updated hybrid_graph_retrieval

```python
def hybrid_graph_retrieval(
    query: str,
    driver: Driver,
    vector_results: list[dict],
    top_k: int = 10,
    collection_name: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """Enhanced hybrid retrieval with scope awareness."""

    # Classify query scope
    scope = classify_query_scope(query)
    community_level = 0 if scope == "local" else 2

    # Get community context at appropriate level
    community_context = retrieve_community_context(
        query,
        level=community_level,
        top_k=GRAPHRAG_TOP_COMMUNITIES,
    )

    # ... existing RRF merge logic ...

    metadata = {
        # ... existing fields ...
        "query_scope": scope,
        "community_level": community_level,
        "community_context": community_context,  # Now includes PageRank + relationships
    }

    return merged_dicts, metadata
```

---

## Execution Order

### Phase 1: Schema Updates (No Data Changes)

1. Add `CommunityRelationship` to `schemas.py`
2. Add `relationships` field to `Community` schema
3. Update `to_dict()` method
4. Update `load_communities()` to parse relationships

### Phase 2: PageRank Implementation

1. Add `run_pagerank()` function to `community.py`
2. Add `write_pagerank_to_neo4j()` function
3. Update `get_community_members()` query to include PageRank
4. Integrate into `detect_and_summarize_communities()` pipeline

### Phase 3: Relationships Storage

1. Update `get_community_relationships()` return type
2. Pass relationships to `Community` constructor
3. Verify JSON output includes relationships

### Phase 4: Hierarchical Communities

1. Add `parse_community_hierarchy()` function
2. Add `get_community_members_by_node_ids()` helper
3. Add `get_community_relationships_by_node_ids()` helper
4. Update `detect_and_summarize_communities()` for multi-level
5. Update `save_communities()` for hierarchical storage
6. Update Weaviate storage (separate collection per level or level filter)

### Phase 5: Query-Time Integration

1. Add `classify_query_scope()` function
2. Update `retrieve_community_context()` with level parameter
3. Update `format_graph_context_for_generation()` for structured data
4. Add `handle_global_query()` map-reduce function
5. Update `hybrid_graph_retrieval()` with scope awareness

### Phase 6: Re-run Pipeline

```bash
# Clear existing communities (keep entities/relationships)
# Option 1: Re-run just community detection
python -m src.stages.run_stage_6b_neo4j --leiden-only

# Option 2: Full re-run with new features
python -m src.stages.run_stage_6b_neo4j --clear
```

---

## Testing Plan

### Unit Tests

1. `test_parse_community_hierarchy()` - Verify level parsing
2. `test_run_pagerank()` - Verify PageRank computation
3. `test_community_relationships_storage()` - Verify JSON serialization
4. `test_classify_query_scope()` - Local vs global classification

### Integration Tests

1. End-to-end: Run full pipeline with new features
2. Query test: Local query uses L0, global uses L2
3. PageRank ordering: High-PageRank entities first in context
4. Relationship access: Structured relationships in generation prompt

### Validation Queries

```python
# Test PageRank is computed
MATCH (e:Entity) WHERE e.pagerank > 0 RETURN count(e)

# Test hierarchy
# Load communities.json and verify levels 0, 1, 2 exist

# Test relationships in JSON
import json
with open("communities.json") as f:
    data = json.load(f)
    assert "relationships" in data["communities"][0]
```

---

## Configuration Reference

**File:** `src/config.py`

```python
# Existing
GRAPHRAG_LEIDEN_RESOLUTION = 1.0
GRAPHRAG_LEIDEN_SEED = 42
GRAPHRAG_MIN_COMMUNITY_SIZE = 3

# New
GRAPHRAG_MAX_COMMUNITY_LEVEL = 2  # Summarize L0, L1, L2
GRAPHRAG_LEVEL_MIN_SIZES = {0: 3, 1: 5, 2: 10}
GRAPHRAG_PAGERANK_DAMPING = 0.85
GRAPHRAG_PAGERANK_ITERATIONS = 20
```

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/graph/schemas.py` | Add `CommunityRelationship`, update `Community` |
| `src/graph/community.py` | Add PageRank, hierarchy parsing, relationship storage |
| `src/graph/query.py` | Add scope classification, level-aware retrieval, map-reduce |
| `src/config.py` | Add hierarchy and PageRank config |
| `src/stages/run_stage_6b_neo4j.py` | Update CLI for new features |

---

## References

- Original Paper: [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
- Current Implementation: `src/graph/` directory
- Memory Bank: `memory-bank/graphrag/`
- Fix History: `memory-bank/graphrag/fix_graphrag.md`

---

*Plan Created: 2026-01-09*
