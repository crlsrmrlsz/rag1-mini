# HyDE: Hypothetical Document Embeddings

> **Paper:** [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) | Gao et al. (CMU) | ACL 2023

Generates a hypothetical answer to the query, then searches for real documents similar to this answer. Bridges the semantic gap between question embeddings and document embeddings.

## TL;DR

Instead of embedding the question directly, HyDE asks an LLM to generate a plausible answer, then embeds *that*. The embedding model's "dense bottleneck" filters out hallucinated details while preserving semantic relevance.

## The Problem

Questions and documents live in different semantic spaces:

```
Question: "What causes stress to affect memory?"
          ↓ embedding
          [query vector - question-like]

Document: "Chronic cortisol elevation impairs hippocampal
           neurogenesis and disrupts memory consolidation..."
          ↓ embedding
          [document vector - statement-like]
```

These vectors may be far apart despite topical relevance because:
- Questions use interrogative structure
- Documents use declarative structure
- Vocabulary differs (scientific vs conversational)

## The Solution

### HyDE Transformation

```
Question: "What causes stress to affect memory?"
          ↓ LLM
Hypothetical: "Stress affects memory through cortisol release.
               The hippocampus, rich in cortisol receptors,
               experiences reduced neurogenesis under chronic
               stress, impairing memory consolidation..."
          ↓ embedding
          [document-like vector - closer to corpus!]
```

The hypothetical may contain hallucinations, but the embedding captures the semantic "shape" of relevant documents.

### Key Insight: Dense Bottleneck

The embedding model was trained on real documents. When you embed a hypothetical:
- **Preserved**: Topics, concepts, semantic relationships
- **Filtered**: Specific wrong facts, hallucinated details

The fixed-dimension embedding is a "bottleneck" that compresses to essence.

## Implementation Details

### Strategy Function

```python
# src/rag_pipeline/retrieval/preprocessing/strategies.py

def hyde_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """HyDE: Generate hypothetical answer, use for retrieval."""
    model = model or PREPROCESSING_MODEL

    # Generate hypothetical answer
    hyde_passage = hyde_prompt(query, model=model)

    return PreprocessedQuery(
        original_query=query,
        search_query=hyde_passage,  # Search with hypothetical!
        hyde_passage=hyde_passage,
        strategy_used="hyde",
    )
```

### HyDE Prompt (Paper-Aligned)

```python
# src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py

HYDE_PROMPT = """Please write a passage from a cognitive science and philosophy knowledge base to answer the question.

Question: {query}

Passage:"""
```

**Why so minimal?**
- Paper finding: Over-specification causes template bias
- Domain hint ("cognitive science and philosophy") guides vocabulary
- No examples, no length constraints — let embedding filter noise

### LLM Call

```python
def hyde_prompt(query: str, model: str) -> str:
    """Generate hypothetical document for query."""
    prompt = HYDE_PROMPT.format(query=query)

    return call_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.7,  # Paper default: creativity for diversity
        max_tokens=200,   # Short passage sufficient
    )
```

### Design Decisions

**Temperature 0.7 (not lower)?**
- Paper uses 0.7 for diverse hypotheticals
- Lower temperature = templated outputs = less embedding diversity
- Trust the encoder to filter noise

**No length constraint?**
- Paper doesn't specify limits
- Encoder bottleneck naturally compresses
- Let LLM generate naturally

**Domain-specific prompt?**
- Paper uses task-specific prompts for specialized corpora
- SciFact: "write a scientific paper passage"
- RAGLab: "cognitive science and philosophy knowledge base"

## When to Use

**Good for:**
- Vague or contextually ambiguous questions
- Zero-shot retrieval (no task-specific training)
- Complex queries requiring semantic understanding
- When query vocabulary differs from corpus

**Limitations:**
- **Knowledge bottleneck**: HyDE struggles when LLM doesn't know the topic
- **Latency**: One LLM call per query (~500ms)
- **Highly specialized domains**: Factual precision may suffer

## Example

**Query**: "Why do we procrastinate on important tasks?"

**Hypothetical** (generated):
```
Procrastination on important tasks often stems from a conflict between
the limbic system's preference for immediate rewards and the prefrontal
cortex's ability to plan for future goals. Temporal discounting causes
distant deadlines to feel less urgent, while task aversion triggers
avoidance behaviors. Self-regulation failure occurs when cognitive
resources are depleted, making it harder to override impulses...
```

**Retrieval**: Finds chunks about temporal discounting, limbic system, self-regulation — even if query didn't use those terms.

## Cost Analysis

- **Model**: `gpt-4o-mini` (~$0.15/1M input, ~$0.60/1M output)
- **Per query**: ~50 input tokens + ~150 output tokens
- **Cost per query**: ~$0.0001

Negligible for evaluation; acceptable for production.

## Results

See [Evaluation Results](../evaluation/results.md) for RAGAS metrics comparing HyDE against none, decomposition, and graphrag.

## Related

- [Query Decomposition](query-decomposition.md) — Alternative for complex queries
- [GraphRAG](graphrag.md) — Entity-based alternative
- [Paper Implementation](https://github.com/texttron/hyde) — Official reference
