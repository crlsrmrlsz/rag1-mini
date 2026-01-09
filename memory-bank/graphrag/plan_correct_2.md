# GraphRAG Improvement Plan: Embedding-Based Query Entity Extraction

**Date:** 2026-01-09
**Status:** Planning
**Priority:** Medium
**Estimated Scope:** ~200 lines of code changes
**Prerequisite:** None (independent of plan_correct_1.md)

---

## Executive Summary

Replace LLM-based query entity extraction with embedding-based semantic similarity, aligning with the Microsoft GraphRAG reference implementation. This eliminates one LLM call per query, reducing latency and cost while maintaining accuracy for explicit entity mentions.

---

## Background: Current vs Reference Implementation

### Discovery

Comparison of RAGLab implementation against Microsoft's reference revealed a fundamental difference in query-time entity extraction:

| Aspect | Microsoft GraphRAG | RAGLab (Current) |
|--------|-------------------|------------------|
| **Method** | Embedding similarity search | LLM prompt |
| **LLM calls at query time** | 0 (for entity extraction) | 1 per query |
| **Latency** | ~50ms (vector search) | ~1-2s (LLM call) |
| **Cost** | $0 | ~$0.001-0.01 per query |

### Microsoft Reference Code

**File:** `graphrag/query/context_builder/entity_extraction.py`

```python
def map_query_to_entities(
    query: str,
    text_embedding_vectorstore: BaseVectorStore,
    text_embedder: EmbeddingModel,
    all_entities_dict: dict[str, Entity],
    k: int = 10,
    oversample_scaler: int = 2,
) -> list[Entity]:
    """Extract entities that match a given query using
    semantic similarity of text embeddings of query and entity descriptions."""

    # 1. Embed the query
    # 2. Vector similarity search against entity description embeddings
    # 3. Return top-k semantically similar entities
```

### RAGLab Current Code

**File:** `src/graph/query.py:85-127`

```python
def extract_query_entities_llm(query: str, model: str) -> list[str]:
    """Extract entities from query using LLM."""
    prompt = GRAPHRAG_QUERY_EXTRACTION_PROMPT.format(
        entity_types=", ".join(entity_types),
        query=query,
    )
    result = call_structured_completion(...)  # LLM call
    return [e.name for e in result.entities]
```

**Current Prompt:** `src/prompts.py:101-119`

```
GRAPHRAG_QUERY_EXTRACTION_PROMPT = """Identify entities mentioned or implied in this query.
Look for: concepts, brain regions, neurotransmitters, philosophers...

Entity types: {entity_types}
Query: {query}

Extract all relevant entities, including:
- Explicitly named entities (e.g., "Sapolsky", "dopamine")
- Implied concepts (e.g., "why we procrastinate" implies "procrastination")
...
"""
```

---

## Problem Statement

1. **Latency:** LLM call adds 1-2 seconds to every query
2. **Cost:** Each query costs $0.001-0.01 for entity extraction alone
3. **Deviation:** Does not match Microsoft's reference implementation approach
4. **Redundancy:** Entity descriptions already exist and could be embedded

---

## Proposed Solution

### Approach: Hybrid with Embedding Primary

1. **Primary:** Embedding-based semantic similarity (fast, cheap)
2. **Fallback:** LLM extraction for complex queries (optional, configurable)

This allows alignment with Microsoft's approach while preserving the ability to handle implied concepts when needed.

---

## Implementation Plan

### Phase 1: Entity Description Embeddings (Indexing Time)

#### Step 1.1: Add Embedding Field to Entity Schema

**File:** `src/graph/schemas.py`

```python
class GraphEntity(BaseModel):
    name: str
    entity_type: str
    description: str
    source_chunk_id: str
    description_embedding: Optional[list[float]] = Field(
        default=None,
        description="Embedding of entity description for query matching"
    )
```

#### Step 1.2: Generate Embeddings During Extraction

**File:** `src/graph/auto_tuning.py`

```python
def extract_book(book_path: Path, model: str, embed_descriptions: bool = True):
    # ... existing extraction code ...

    if embed_descriptions:
        # Batch embed all entity descriptions
        descriptions = [e["description"] for e in all_entities if e.get("description")]
        if descriptions:
            embeddings = embed_texts(descriptions)
            for i, entity in enumerate(all_entities):
                if entity.get("description"):
                    entity["description_embedding"] = embeddings[i]
```

#### Step 1.3: Store Embeddings in Neo4j (Optional) or Weaviate

**Option A: Neo4j (simpler, but no native vector search)**

```python
# Store as property (for later export to vector store)
e.description_embedding = $embedding
```

**Option B: Weaviate Entity Collection (recommended)**

Create a new Weaviate collection for entity descriptions:

**File:** `src/rag_pipeline/indexing/weaviate_client.py`

```python
def create_entity_collection(client: WeaviateClient, collection_name: str):
    """Create collection for entity description embeddings."""
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="entity_name", data_type=DataType.TEXT),
            Property(name="normalized_name", data_type=DataType.TEXT),
            Property(name="entity_type", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
        ],
    )


def upload_entity_embeddings(
    client: WeaviateClient,
    collection_name: str,
    entities: list[dict],
) -> int:
    """Upload entity descriptions with embeddings to Weaviate."""
    collection = client.collections.get(collection_name)

    with collection.batch.dynamic() as batch:
        for entity in entities:
            if entity.get("description_embedding"):
                batch.add_object(
                    properties={
                        "entity_name": entity["name"],
                        "normalized_name": entity["normalized_name"],
                        "entity_type": entity["entity_type"],
                        "description": entity["description"],
                    },
                    vector=entity["description_embedding"],
                )

    return len(entities)
```

#### Step 1.4: Configuration

**File:** `src/config.py`

```python
# Entity embedding collection
def get_entity_collection_name() -> str:
    return f"Entity_{CHUNKING_STRATEGY}{MAX_CHUNK_TOKENS}_v{COLLECTION_VERSION}"

# Query entity extraction settings
GRAPHRAG_ENTITY_EXTRACTION_METHOD = "embedding"  # "embedding" or "llm"
GRAPHRAG_ENTITY_TOP_K = 10  # Top entities to retrieve
GRAPHRAG_ENTITY_OVERSAMPLE = 2  # Oversample factor for filtering
GRAPHRAG_ENTITY_MIN_SIMILARITY = 0.3  # Minimum similarity threshold
```

---

### Phase 2: Query-Time Entity Extraction (Embedding-Based)

#### Step 2.1: New Function for Embedding-Based Extraction

**File:** `src/graph/query.py`

```python
def extract_query_entities_embedding(
    query: str,
    top_k: int = GRAPHRAG_ENTITY_TOP_K,
    min_similarity: float = GRAPHRAG_ENTITY_MIN_SIMILARITY,
) -> list[dict]:
    """Extract entities matching query using embedding similarity.

    Follows Microsoft GraphRAG reference implementation approach.

    Args:
        query: User query string.
        top_k: Number of top entities to return.
        min_similarity: Minimum similarity threshold.

    Returns:
        List of entity dicts with name, type, description, similarity.

    Example:
        >>> entities = extract_query_entities_embedding("How does dopamine affect mood?")
        >>> for e in entities:
        ...     print(e["entity_name"], e["similarity"])
        dopamine 0.87
        mood 0.82
        serotonin 0.71
    """
    from src.rag_pipeline.embedding.embedder import embed_texts
    from src.rag_pipeline.indexing.weaviate_client import get_client

    # Embed the query
    query_embedding = embed_texts([query])[0]

    # Search entity collection
    collection_name = get_entity_collection_name()
    client = get_client()

    try:
        collection = client.collections.get(collection_name)

        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k * GRAPHRAG_ENTITY_OVERSAMPLE,  # Oversample for filtering
            return_metadata=MetadataQuery(distance=True),
        )

        entities = []
        for obj in response.objects:
            similarity = 1 - obj.metadata.distance  # Convert distance to similarity
            if similarity >= min_similarity:
                entities.append({
                    "entity_name": obj.properties["entity_name"],
                    "normalized_name": obj.properties["normalized_name"],
                    "entity_type": obj.properties["entity_type"],
                    "description": obj.properties["description"],
                    "similarity": similarity,
                })

        # Sort by similarity and take top_k
        entities.sort(key=lambda x: x["similarity"], reverse=True)
        entities = entities[:top_k]

        logger.info(f"Embedding extraction found {len(entities)} entities")
        return entities

    finally:
        client.close()
```

#### Step 2.2: Update Main Extraction Function

**File:** `src/graph/query.py`

```python
def extract_query_entities(
    query: str,
    driver: Optional[Driver] = None,
    method: str = GRAPHRAG_ENTITY_EXTRACTION_METHOD,
) -> list[str]:
    """Extract entity mentions from query.

    Args:
        query: User query string.
        driver: Optional Neo4j driver for validation.
        method: "embedding" (default) or "llm".

    Returns:
        List of entity names found in query.
    """
    if method == "embedding":
        # Primary: Embedding-based (Microsoft reference approach)
        entities = extract_query_entities_embedding(query)
        entity_names = [e["entity_name"] for e in entities]
        logger.info(f"Embedding extracted: {entity_names}")

    elif method == "llm":
        # Alternative: LLM-based (original RAGLab approach)
        entity_names = extract_query_entities_llm(query)
        logger.info(f"LLM extracted: {entity_names}")

    else:
        raise ValueError(f"Unknown extraction method: {method}")

    # Validate against Neo4j if driver provided
    if driver and entity_names:
        db_entities = find_entities_by_names(driver, entity_names)
        validated = [e["name"] for e in db_entities]
        logger.info(f"Validated in Neo4j: {validated}")
        return validated

    return entity_names
```

#### Step 2.3: Keep LLM as Optional Fallback

The existing `extract_query_entities_llm()` function remains unchanged, available via `method="llm"` parameter or as automatic fallback when embedding search returns no results.

```python
def extract_query_entities(query: str, driver: Optional[Driver] = None, method: str = "embedding"):
    if method == "embedding":
        entities = extract_query_entities_embedding(query)

        # Fallback to LLM if no entities found
        if not entities:
            logger.info("No entities from embedding, falling back to LLM")
            return extract_query_entities_llm(query)

        return [e["entity_name"] for e in entities]
    # ...
```

---

### Phase 3: Stage Runner Updates

#### Step 3.1: Add Entity Embedding to Stage 6b

**File:** `src/stages/run_stage_6b_neo4j.py`

Add new CLI flag and phase:

```python
@click.option("--embed-entities", is_flag=True, help="Generate entity description embeddings")
def main(..., embed_entities: bool):
    # ... existing upload code ...

    if embed_entities:
        logger.info("Phase 4: Generating entity description embeddings")

        # Load entities from extraction results
        entities = results["entities"]

        # Filter entities with descriptions
        entities_with_desc = [e for e in entities if e.get("description")]

        # Batch embed descriptions
        descriptions = [e["description"] for e in entities_with_desc]
        embeddings = embed_texts(descriptions)

        for entity, embedding in zip(entities_with_desc, embeddings):
            entity["description_embedding"] = embedding

        # Upload to Weaviate
        collection_name = get_entity_collection_name()
        client = get_weaviate_client()

        if not client.collections.exists(collection_name):
            create_entity_collection(client, collection_name)

        count = upload_entity_embeddings(client, collection_name, entities_with_desc)
        logger.info(f"Uploaded {count} entity embeddings to {collection_name}")
```

---

## Execution Order

### One-Time Indexing (run once)

```bash
# Option 1: Add to existing Stage 6b run
python -m src.stages.run_stage_6b_neo4j --embed-entities

# Option 2: Standalone script (if keeping separate)
python -m src.stages.run_entity_embeddings
```

### Query-Time (automatic)

No changes needed - `extract_query_entities()` will use embedding method by default.

---

## Testing Plan

### Unit Tests

```python
def test_extract_query_entities_embedding():
    """Test embedding-based entity extraction."""
    entities = extract_query_entities_embedding("How does dopamine affect motivation?")

    assert len(entities) > 0
    assert any("dopamine" in e["entity_name"].lower() for e in entities)
    assert all(e["similarity"] >= GRAPHRAG_ENTITY_MIN_SIMILARITY for e in entities)


def test_extract_query_entities_fallback():
    """Test fallback to LLM when embedding returns nothing."""
    # Query with implied concept (unlikely to match entity descriptions)
    entities = extract_query_entities("Why do we procrastinate?", method="embedding")

    # Should either find relevant entities or fallback to LLM
    assert len(entities) >= 0  # May be empty if no fallback triggered
```

### Performance Comparison

```python
import time

def benchmark_extraction_methods(queries: list[str]):
    """Compare latency of embedding vs LLM extraction."""

    results = {"embedding": [], "llm": []}

    for query in queries:
        # Embedding method
        start = time.time()
        extract_query_entities(query, method="embedding")
        results["embedding"].append(time.time() - start)

        # LLM method
        start = time.time()
        extract_query_entities(query, method="llm")
        results["llm"].append(time.time() - start)

    print(f"Embedding avg: {sum(results['embedding'])/len(queries)*1000:.0f}ms")
    print(f"LLM avg: {sum(results['llm'])/len(queries)*1000:.0f}ms")
```

---

## Trade-offs

### Advantages of Embedding Approach

| Benefit | Impact |
|---------|--------|
| **Latency** | ~50ms vs ~1-2s (20-40x faster) |
| **Cost** | $0 vs $0.001-0.01 per query |
| **Consistency** | Deterministic results |
| **Alignment** | Matches Microsoft reference |

### Disadvantages / Limitations

| Limitation | Mitigation |
|------------|------------|
| Misses implied concepts | LLM fallback for complex queries |
| Requires description quality | Entity descriptions from extraction are usually good |
| Additional indexing step | One-time cost, ~10min for typical corpus |
| Storage overhead | ~10MB for entity embeddings (negligible) |

### When to Use LLM Fallback

- Query contains implied concepts ("why we procrastinate" → procrastination)
- Embedding returns zero results
- Query is phrased as a question about abstract concepts
- User explicitly requests deeper extraction

---

## Configuration Reference

**File:** `src/config.py`

```python
# Entity extraction method
GRAPHRAG_ENTITY_EXTRACTION_METHOD = "embedding"  # "embedding" or "llm"

# Embedding extraction parameters
GRAPHRAG_ENTITY_TOP_K = 10           # Top entities to retrieve
GRAPHRAG_ENTITY_OVERSAMPLE = 2       # Oversample factor
GRAPHRAG_ENTITY_MIN_SIMILARITY = 0.3 # Minimum similarity threshold

# Collection naming
def get_entity_collection_name() -> str:
    return f"Entity_{CHUNKING_STRATEGY}{MAX_CHUNK_TOKENS}_v{COLLECTION_VERSION}"
```

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/graph/schemas.py` | Add `description_embedding` field to GraphEntity |
| `src/graph/query.py` | Add `extract_query_entities_embedding()`, update `extract_query_entities()` |
| `src/rag_pipeline/indexing/weaviate_client.py` | Add `create_entity_collection()`, `upload_entity_embeddings()` |
| `src/stages/run_stage_6b_neo4j.py` | Add `--embed-entities` flag |
| `src/config.py` | Add entity extraction config parameters |

---

## References

- [Microsoft GraphRAG entity_extraction.py](https://github.com/microsoft/graphrag/blob/main/graphrag/query/context_builder/entity_extraction.py)
- [GraphRAG Local Search Documentation](https://microsoft.github.io/graphrag/query/local_search/)
- [GraphRAG-Prompts Repository](https://github.com/langgptai/GraphRAG-Prompts)
- Current implementation: `src/graph/query.py:85-127`

---

*Plan Created: 2026-01-09*
