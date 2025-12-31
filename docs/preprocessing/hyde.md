# HyDE: Hypothetical Document Embeddings

> **Paper:** [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) | Gao et al. (CMU) | ACL 2023

Generates hypothetical answers to the query, then searches for real documents similar to these answers. Bridges the semantic gap between question embeddings and document embeddings.

**December 2024 Update:** Implements K=2 multi-hypothetical generation with embedding averaging (configurable via `HYDE_K` in config.py).

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
    """HyDE: Generate K hypotheticals, average embeddings for retrieval."""
    model = model or PREPROCESSING_MODEL

    # Generate K hypothetical answers (configurable in config.py)
    hyde_passages = hyde_prompt(query, model=model, k=HYDE_K)

    return PreprocessedQuery(
        original_query=query,
        search_query=hyde_passages[0],  # First for backward compat
        hyde_passage=hyde_passages[0],  # Keep first for logging
        generated_queries=[{"type": "hyde", "query": p} for p in hyde_passages],
        strategy_used="hyde",
    )
```

At retrieval time, all K passages in `generated_queries` are embedded and averaged.

### HyDE Prompt (Paper-Aligned + Corpus Hints)

```python
# src/rag_pipeline/retrieval/preprocessing/query_preprocessing.py

HYDE_PROMPT = """Please write a short passage drawing on insights from brain science and classical philosophy (Stoicism, Taoism, Confucianism, Schopenhauer, Gracian) to answer the question.

Question: {query}

Passage:"""
```

**Design Rationale:**
- **"Drawing on insights from..."** — Requests cross-domain synthesis for our mixed corpus
- **Parenthetical tradition hints** — "(Stoicism, Taoism, Confucianism, Schopenhauer, Gracian)" provides specific corpus cues without vocabulary lists
- **Covers all 10 philosophy books** in the corpus
- Paper finding: Over-specification causes template bias, but domain-specific hints improve retrieval

### Multi-Hypothetical Generation (Configurable K)

```python
def hyde_prompt(query: str, model: str, k: int = HYDE_K) -> List[str]:
    """Generate k hypothetical documents for query.

    Multiple hypotheticals improve retrieval robustness by covering
    diverse phrasings and perspectives. Embeddings are averaged downstream.
    """
    prompt = HYDE_PROMPT.format(query=query)
    passages = []

    for _ in range(k):
        response = call_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,  # Paper default for diversity
        )
        passages.append(response.strip())

    return passages
```

At retrieval time:
1. Embed all K passages using the same embedding model
2. Average the embedding vectors (element-wise mean)
3. Use averaged vector for hybrid search
4. Original query still used for BM25 keyword matching

**Why multiple hypotheticals?**
- Paper finding: Multiple hypotheticals improve retrieval robustness
- Different passages capture different phrasings/perspectives
- Averaging creates a more centered representation in embedding space
- RAGLab uses K=2 by default (configurable via `HYDE_K` in config.py)

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

With K=2 hypotheticals (default):

- **Model**: `gpt-4o-mini` (~$0.15/1M input, ~$0.60/1M output)
- **Per query**: 2 x (~50 input tokens + ~150 output tokens) = ~400 tokens
- **LLM cost per query**: ~$0.0002
- **Embedding cost**: 2 passages embedded instead of 1

**Total per query**: ~$0.0004 (negligible for evaluation)

Trade-off: 2x latency for LLM calls, but more robust retrieval. Increase `HYDE_K` in config.py for better robustness at higher cost.

## Results

See [Evaluation Results](../evaluation/results.md) for RAGAS metrics comparing HyDE against none, decomposition, and graphrag.

## Related

- [Query Decomposition](query-decomposition.md) — Alternative for complex queries
- [GraphRAG](graphrag.md) — Entity-based alternative
- [Paper Implementation](https://github.com/texttron/hyde) — Official reference
