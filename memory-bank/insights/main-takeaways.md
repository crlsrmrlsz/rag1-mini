# RAG Evaluation: Main Takeaways

**Date:** 2026-01-04
**Source:** Comprehensive evaluation of 102 configurations across 15 questions (5 single-concept, 10 cross-domain)
**Analysis Method:** Deep logical analysis backed by statistical evidence

---

## 1. Recall Matters More Than Precision for Answer Quality

**Statistical Evidence:**
| Strategy | Precision | Recall | Answer Correctness |
|----------|-----------|--------|-------------------|
| Semantic 0.3 | **73.4%** (1st) | 93.3% | 54.1% (4th) |
| Contextual | 71.7% (2nd) | **96.3%** (1st) | **59.1%** (1st) |
| GraphRAG | 66.4% (4th) | **97.5%** (1st) | **59.5%** (1st) |

**Logical Explanation:**
The RAG pipeline has an **information-theoretic asymmetry**: the LLM generator can filter irrelevant context (low precision is recoverable) but cannot invent missing information (low recall is unrecoverable). When relevant facts are absent from the context window, even the best LLM cannot hallucinate them correctly. High recall ensures all necessary pieces reach the generator; low precision just adds noise the generator can ignore.

**Design Principle:** *Optimize retrieval for coverage, not purity. Trust the generator to filter; don't trust it to invent.*

---

## 2. Cross-Domain Queries Expose the Fundamental Limits of Vector Similarity

**Statistical Evidence:**
| Strategy | Single-Concept Recall | Cross-Domain Recall | Delta |
|----------|----------------------|---------------------|-------|
| HyDE | 89.3% | 78.8% | **-10.5%** (best) |
| GraphRAG | 97.5% | 76.1% | -21.4% |
| None | 92.3% | 70.5% | -21.8% |
| Decomposition | 96.0% | 65.6% | **-30.4%** (worst) |

**Logical Explanation:**
Vector similarity operates from a **single query point** in embedding space. Cross-domain queries (requiring synthesis from 4-7 books across neuroscience AND philosophy) need to find chunks near **multiple conceptual clusters simultaneously**. The query embedding becomes a centroid equidistant from all relevant clusters—optimally far from any specific chunk.

- **HyDE works** because the hypothetical answer **pre-synthesizes** the cross-domain bridge, creating an embedding that spans both domains
- **Decomposition fails** because it fragments the synthesis requirement into isolated sub-queries, then RRF-merges results. Chunks that genuinely bridge domains rank moderately for each sub-query and get demoted

**Design Principle:** *For cross-domain synthesis, use HyDE or GraphRAG. Avoid decomposition—it destroys the very integration the query requires.*

---

## 3. The Pipeline Has Clear Variance Ownership: Search Type → Precision, Chunking → Recall

**Statistical Evidence:**
| Metric (Cross-Domain) | Most Influential Dimension | Variance Explained |
|-----------------------|---------------------------|-------------------|
| Context Precision | **Search Type** | 92% |
| Context Recall | **Chunking (Collection)** | 65% |
| Answer Correctness | Search Type | 22% |
| Relevancy | Chunking | 28% |

Preprocessing strategy has moderate (7-24%) but never dominant influence on any metric.

**Logical Explanation:**
- **Search type** is the ranking function that determines which chunks appear in top-k. It directly controls what gets selected → dominates precision
- **Chunking strategy** defines the universe of addressable units. A passage can only be retrieved if it exists as a discrete chunk → dominates recall (sets the ceiling of what's retrievable)

This is analogous to library organization: chunking is how books are shelved (what logical units exist); search type is the catalog system (how you find them).

**Design Principle:** *To improve precision, tune search type (semantic > hybrid > keyword). To improve recall, tune chunking (contextual > section > semantic).*

---

## 4. Knowledge Graphs Encode Relationships That Vectors Cannot

**Statistical Evidence:**
| Metric | GraphRAG | HyDE | Baseline (none) |
|--------|----------|------|-----------------|
| Single-Concept AC | **59.5%** | 55.6% | 56.3% |
| Cross-Domain AC | **50.1%** | 47.3% | 47.7% |
| Cross-Domain Recall | 76.1% | **78.8%** | 70.5% |

GraphRAG achieves **+5.7%** answer correctness over baseline despite having lower recall than HyDE.

**Logical Explanation:**
Vector embeddings encode **what text is about** but not **how concepts relate**. The knowledge graph captures three types of information invisible to cosine similarity:

1. **Typed relationships**: `dopamine --[MODULATES]--> reward` encodes causality that "dopamine and reward are similar" cannot
2. **Transitive connections**: 2-hop paths find intermediate concepts (childhood_trauma → amygdala → aggression) that link disconnected chunks
3. **Community summaries**: Leiden clustering pre-computes thematic clusters with LLM-generated summaries, capturing cross-document patterns

When a cross-domain question asks "how do neuroscience and Stoicism address free will?", the graph **already knows** that Marcus Aurelius and prefrontal cortex both connect to determinism—it doesn't need to discover this via embedding proximity.

**Design Principle:** *For questions about how concepts relate (not just what they mean), GraphRAG's +5% improvement justifies its infrastructure overhead.*

---

## 5. Contextual Chunking Wins by Disambiguating Chunk Identity

**Statistical Evidence:**
| Chunking Strategy | Single-Concept AC | Cross-Domain AC | Single Recall |
|-------------------|-------------------|-----------------|---------------|
| **Contextual** | **59.1%** | **48.8%** | **96.3%** |
| RAPTOR | 57.9% | 48.4% | 96.1% |
| Section | 57.6% | 47.9% | 92.9% |
| Semantic 0.3 | 54.1% | 48.0% | 93.3% |
| Semantic 0.75 | 50.8% | 45.6% | 86.1% |

**Logical Explanation:**
Contextual chunking prepends a 2-3 sentence LLM-generated summary: *"This chunk from Chapter 5 of Behave discusses the amygdala's response to threat..."*

This transforms the chunk from "text that contains these words" to "text that IS ABOUT this topic." The embedding model can now understand the chunk's semantic role, not just its vocabulary. When a query uses different terminology than the source text, the contextual prefix provides the semantic anchor that bridges the vocabulary gap.

The synergy with GraphRAG (+5% on top) occurs because they operate on orthogonal dimensions: contextual provides intra-document clarity; GraphRAG provides inter-document connections.

**Design Principle:** *Contextual chunking is the highest-ROI improvement: one-time LLM cost at index time for consistent answer quality gains across all query types.*

---

## Quick Reference: Best Configurations

### For Single-Concept Queries
```
Chunking:       Contextual
Preprocessing:  GraphRAG
Search Type:    Hybrid
Expected AC:    ~62%
```

### For Cross-Domain Queries
```
Chunking:       Contextual (or RAPTOR for faithfulness)
Preprocessing:  GraphRAG (or HyDE for simpler infrastructure)
Search Type:    Semantic
Expected AC:    ~52%
```

### Universal Recommendation
```
Chunking:       Contextual
Preprocessing:  GraphRAG
Search Type:    Hybrid (or semantic if cross-domain heavy)
```

---

## Summary Table

| Insight | Statistical Backing | Logical Mechanism |
|---------|---------------------|-------------------|
| Recall > Precision | Best precision (73.4%) → 4th correctness; Best recall (96.3%) → 1st correctness | Generator can filter noise but cannot invent missing info |
| Cross-domain exposes vector limits | 17-30% recall drop; HyDE -10.5%, Decomposition -30.4% | Single query point cannot reach multiple clusters; HyDE pre-synthesizes bridge |
| Search → Precision, Chunking → Recall | 92% precision variance from search; 65% recall variance from chunking | Ranking determines selection; chunking defines what's addressable |
| Graphs encode relationships | +5.7% correctness despite lower recall than HyDE | Typed edges, 2-hop paths, and community summaries capture causality |
| Contextual chunking adds identity | Best correctness on both query types (59.1%, 48.8%) | LLM prefix tells embedding model what chunk IS ABOUT, not just what words it has |

---

## Anti-Patterns to Avoid

| Configuration | Problem | Evidence |
|---------------|---------|----------|
| Decomposition + Cross-domain | Fragments synthesis requirement | -30.4% recall drop |
| Semantic 0.75 chunking | Loose threshold creates incoherent chunks | Worst on all metrics |
| Keyword search alone | Vocabulary mismatch kills precision | 10+ points below hybrid/semantic |
| Optimizing for precision over recall | High precision doesn't translate to answer quality | Semantic 0.3: best precision, 4th correctness |
| **GraphRAG + Reranking** | Conflicting objectives destroy graph-derived diversity | **-26.7% AC degradation** |
| **HyDE + Reranking** | Semantic drift compounds | -8% to -16% AC degradation |

---

## 6. Reranking: Precision vs Coverage Tradeoff (Updated 2026-01-05)

**Statistical Evidence:**
| Configuration | Context Precision | Answer Correctness | Delta AC |
|---------------|-------------------|-------------------|----------|
| Contextual + GraphRAG + Hybrid (baseline) | 88.3% | **60.2%** | - |
| Contextual + GraphRAG + Hybrid + Rerank | **94.2%** | 44.1% | **-26.7%** |
| Section + HyDE + Semantic (baseline) | 87.6% | **60.6%** | - |
| Section + HyDE + Semantic + Rerank | **90.1%** | 54.3% | **-10.4%** |

**Logical Explanation:**
Cross-encoder reranking optimizes for **query-chunk textual similarity**. This conflicts with preprocessing strategies that retrieve chunks based on different criteria:

- **GraphRAG** retrieves chunks via **graph structure** (entity relationships, community membership). These chunks are topically important but may use different vocabulary than the query. Reranking demotes them, destroying the diversity GraphRAG created.

- **HyDE** retrieves chunks similar to **hypothetical answers**, not the original query. Reranking then evaluates against the original query, penalizing chunks that matched the hypotheticals.

The fundamental problem: **reranking optimizes for precision at the cost of coverage**. The generator can filter noise (low precision is recoverable) but cannot invent missing information (low recall is unrecoverable).

**Design Principle:** *Avoid reranking with GraphRAG and HyDE. If reranking is required, use semantic search (alpha=1.0) which aligns with the cross-encoder's semantic paradigm.*

---

*Generated: 2026-01-04, Updated: 2026-01-05 | Analysis: 4 parallel subagents with deep logical reasoning + reranking evaluation*
