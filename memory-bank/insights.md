# RAG Evaluation Analysis: Key Insights from 102 Configuration Combinations

**Analysis Date:** 2026-01-06
**Data Sources:**
- `comprehensive_20260101_164236.json` - 102 grid search configurations (no reranking)
- `eval_20260104_*.json` - 5 single-run evaluations with reranking

**Test Set:** 15 curated questions (5 single-concept, 10 cross-domain)

---

## Insight 1: Preprocessing Strategy Has the Largest Impact on Cross-Domain Retrieval

**Statistical Evidence:**

| Strategy | Cross-Domain Recall | Degradation vs Single | N |
|----------|--------------------|-----------------------|---|
| **HyDE** | **78.8%** | **-10.5%** | 30 |
| GraphRAG | 76.1% | -21.4% | 12 |
| None | 70.5% | -21.8% | 30 |
| Decomposition | 65.6% | **-30.4%** | 30 |

**Cohen's d (HyDE vs Decomposition) = 1.31 (large effect)**

**Interpretation:** HyDE generates hypothetical answer passages that contain domain-appropriate vocabulary, bridging the semantic gap between question phrasing and document content. Decomposition fragments cross-domain questions into sub-queries that each target one domain—but RRF merging cannot reconstruct the synthesis the original question required. The 13.2 percentage point difference (78.8% vs 65.6%) represents retrieving ~1.3 more relevant ground-truth passages on average per question.

**Recommendation for cross-domain queries:** Use HyDE. Avoid decomposition unless questions are purely multi-step within a single domain.

---

## Insight 2: Chunking Strategy Determines the Ceiling for Cross-Domain Performance

**Statistical Evidence:**

| Collection | Cross-Domain Recall | Degradation vs Single | N |
|------------|--------------------|-----------------------|---|
| **Contextual** | **79.4%** | -16.8% | 24 |
| Raptor | 76.4% | -19.7% | 18 |
| Section | 76.3% | -16.6% | 24 |
| Semantic_0.3 | 69.3% | -24.0% | 18 |
| Semantic_0.75 | 55.6% | **-30.5%** | 18 |

**Cohen's d (Contextual vs Semantic_0.75) = 2.66 (very large effect)**

**Interpretation:** Contextual chunking prepends LLM-generated summaries to each chunk, disambiguating "what this chunk is about." This helps embeddings correctly associate abstract philosophical discussions with their topics. Semantic chunking with loose threshold (0.75) creates weakly-coherent chunks that fall apart under cross-domain retrieval. The 23.8 percentage point gap is the largest measured effect—chunking choice matters more than any runtime parameter.

**Recommendation:** Contextual chunking is essential for mixed-domain corpora. Semantic chunking with loose thresholds should be avoided.

---

## Insight 3: Retrieval Depth (top_k) Matters More for Cross-Domain Than Single-Concept

**Statistical Evidence:**

| Top_K | Single-Concept Recall | Cross-Domain Recall | Cross-Domain Degradation |
|-------|----------------------|--------------------|--------------------------|
| 10 | 91.8% | 68.2% | -23.5% |
| **20** | 94.5% | **76.1%** | -18.4% |

**Cohen's d (top_k=20 vs 10) = 0.70 (medium effect)**

**Interpretation:** Cross-domain questions require evidence from multiple sources (avg. 4-6 expected books per question). With top_k=10, only ~68% of ground truth is retrieved because relevant passages from secondary sources get cut off. Increasing to top_k=20 captures more diverse sources. Single-concept questions already achieve near-ceiling recall (92-95%) because they need only one source.

**Recommendation:** For cross-domain synthesis, use top_k>=20. The computational cost is justified by the 7.9 percentage point recall improvement.

---

## Insight 4: Answer Correctness Remains the Persistent Bottleneck (~48% Cross-Domain)

**Statistical Evidence:**

| Metric | Single-Concept | Cross-Domain | Best Observed |
|--------|---------------|--------------|---------------|
| Context Recall | 93.1% +/-8.7% | 72.2% +/-11.9% | 89.4% |
| **Answer Correctness** | 56.2% +/-7.0% | **47.8% +/-5.1%** | **59.5%** |

**Variance Analysis:**
- Context recall range: 0.344 - 0.894 (55 pt spread)
- Answer correctness range: 0.353 - 0.595 (24 pt spread)

**Interpretation:** Even when retrieval improves dramatically (top configurations achieve 89% cross-domain recall), answer correctness plateaus around 48-50%. The narrow variance in correctness (5.1% std for cross-domain) suggests the generation LLM, not retrieval, is the limiting factor. This is expected: RAGAS answer_correctness compares generated answers to reference paragraphs spanning 4-7 books—a synthesis task that exceeds current model capabilities.

**Recommendation:** Further optimization should focus on generation prompts, answer synthesis strategies, or model selection rather than retrieval tuning. Retrieval improvements alone cannot push correctness beyond ~60%.

---

## Summary of Effect Sizes

| Comparison | Cohen's d | Interpretation |
|------------|-----------|----------------|
| HyDE vs Decomposition | 1.31 | **Large**: Preprocessing choice dominates |
| Contextual vs Semantic_0.75 | 2.66 | **Very Large**: Chunking is foundational |
| Top_k 20 vs 10 | 0.70 | **Medium**: Worth optimizing |
| Keyword vs Semantic | 0.02 | **Negligible**: No significant difference |

---

## Best Configurations by Metric

### Best 5 for Cross-Domain Context Recall

| Rank | Collection | Strategy | Search | TopK | Cross Recall | Cross Correctness |
|------|------------|----------|--------|------|--------------|-------------------|
| 1 | contextual | hyde | semantic | 10 | 89.5% | 55.3% |
| 2 | raptor | none | hybrid | 20 | 89.3% | 55.3% |
| 3 | contextual | graphrag | keyword | 20 | 88.6% | 48.4% |
| 4 | contextual | hyde | semantic | 20 | 87.7% | 45.8% |
| 5 | contextual | hyde | hybrid | 20 | 87.7% | 47.0% |

### Best 5 for Cross-Domain Answer Correctness

| Rank | Collection | Strategy | Search | TopK | Cross Correctness | Cross Recall |
|------|------------|----------|--------|------|-------------------|--------------|
| 1 | raptor | none | semantic | 20 | 59.5% | 79.7% |
| 2 | contextual | graphrag | semantic | 10 | 59.4% | 73.3% |
| 3 | section | hyde | semantic | 20 | 57.5% | 83.8% |
| 4 | semantic_0_75 | decomposition | semantic | 20 | 56.5% | 46.7% |
| 5 | raptor | none | semantic | 10 | 56.4% | 69.9% |

### Best 5 for Single-Concept Context Recall

| Rank | Collection | Strategy | Search | TopK | Single Recall | Single Correctness |
|------|------------|----------|--------|------|---------------|-------------------|
| 1 | contextual | graphrag | keyword | 20 | 100.0% | 67.5% |
| 2 | contextual | graphrag | hybrid | 20 | 100.0% | 68.6% |
| 3 | contextual | none | semantic | 20 | 100.0% | 58.7% |
| 4 | raptor | hyde | hybrid | 10 | 100.0% | 48.9% |
| 5 | section | graphrag | keyword | 20 | 100.0% | 55.3% |

### Best 5 for Single-Concept Answer Correctness

| Rank | Collection | Strategy | Search | TopK | Single Correctness | Single Recall |
|------|------------|----------|--------|------|-------------------|---------------|
| 1 | contextual | decomposition | keyword | 20 | 72.3% | 100.0% |
| 2 | contextual | graphrag | hybrid | 20 | 68.6% | 100.0% |
| 3 | raptor | decomposition | keyword | 10 | 68.2% | 100.0% |
| 4 | contextual | graphrag | keyword | 20 | 67.5% | 100.0% |
| 5 | semantic_0_75 | hyde | keyword | 10 | 66.8% | 60.0% |

**Search type legend:** keyword (alpha=0.0, BM25 only), hybrid (alpha=0.5, balanced), semantic (alpha=1.0, vector only)

---

## Interaction Effects

### Preprocessing Strategy x Search Type (Cross-Domain Recall)

| Strategy | Keyword | Semantic/Hybrid | Delta |
|----------|---------|-----------------|-------|
| none | 69.6% | 71.0% | +1.4% |
| hyde | 79.5% | 78.5% | -0.9% |
| decomposition | 66.2% | 65.3% | -0.9% |
| graphrag | 76.9% | 75.7% | -1.2% |

**Observation:** Search type (keyword vs semantic) has negligible effect on cross-domain recall. The BM25 component does not meaningfully help cross-domain retrieval.

### Collection x Strategy (Cross-Domain Recall)

| Collection | none | hyde | decomp | graphrag |
|------------|------|------|--------|----------|
| contextual | 80.9% | 85.4% | 73.9% | 77.5% |
| raptor | 79.5% | 82.9% | 66.8% | - |
| section | 76.7% | 82.4% | 71.5% | 74.7% |
| semantic_0_3 | 66.3% | 76.4% | 65.2% | - |
| semantic_0_75 | 49.2% | 67.0% | 50.5% | - |

**Observation:** HyDE consistently improves cross-domain recall across all chunking strategies (avg. +8.4 pp). The improvement is largest for weaker chunking strategies (semantic_0_75: +17.8 pp).

---

## Problematic Questions

### Consistently Low Recall Across All Configurations

| Question ID | Mean Recall | Issue |
|-------------|-------------|-------|
| cross_freewill_01 | 37.5% | Requires integration of neuroscience determinism + Stoic philosophy on control |
| cross_selfcontrol_01 | 60-100% | High variance; depends on configuration |
| cross_procrastination_01 | 50-100% | High variance |
| cross_empathy_01 | 66.7-75% | Moderate variance |

**cross_freewill_01 Analysis:** This question ("What determines human choices and actions, and to what extent can we control our own decisions?") requires ground truth from 6 books spanning neuroscience (Sapolsky's Determined, Gazzaniga's Cognitive Neuroscience) and philosophy (Epictetus, Marcus Aurelius, Tao Te Ching). The 37.5% recall ceiling across ALL configurations suggests either:
1. Corpus gaps (missing key passages)
2. Fundamental embedding space distance between neuroscience and philosophy vocabularies

---

## Limitations

1. **Small question set (n=15)**: Per-question metrics have high variance; aggregate statistics are more reliable than individual question analysis.
2. **Specific corpus**: Results reflect a neuroscience + philosophy mixed corpus; generalization to other domains requires validation.
3. **No reranking in grid search**: Reranking evaluations were single-run; comparison across configurations requires caution.
4. **One question consistently fails**: `cross_freewill_01` achieves only 37.5% recall across ALL configurations.

---

## Actionable Recommendations (Priority Order)

1. **Always use Contextual chunking** for mixed-domain corpora (Cohen's d = 2.66)
2. **Use HyDE preprocessing** for cross-domain queries (Cohen's d = 1.31)
3. **Set top_k >= 20** for cross-domain retrieval (Cohen's d = 0.70)
4. **Avoid Decomposition** for questions requiring multi-source synthesis
5. **Accept ~50% correctness ceiling** as a generation-side limitation

---

## Key Takeaway

Foundational decisions (chunking strategy) have 2-4x larger effect sizes than runtime parameters (preprocessing, top_k). For a new RAG system, invest effort in chunking design first, then optimize preprocessing strategy second. Retrieval tuning cannot compensate for poor chunking choices.

---

## GraphRAG Deep Dive

GraphRAG was tested with 12 configurations (contextual and section collections only, as it requires pre-built knowledge graphs).

### GraphRAG Achieves Highest Answer Correctness

| Strategy | Single-Concept Corr | Cross-Domain Corr | N |
|----------|--------------------|--------------------|---|
| **GraphRAG** | **59.5% (+/-4.6%)** | **50.1% (+/-4.1%)** | 12 |
| None | 56.3% (+/-5.1%) | 47.7% (+/-5.6%) | 30 |
| HyDE | 55.6% (+/-8.6%) | 47.3% (+/-5.3%) | 30 |
| Decomposition | 55.4% (+/-7.4%) | 47.5% (+/-4.7%) | 30 |

**Key Finding:** GraphRAG improves answer correctness by +2.4 percentage points (50.1% vs 47.7% baseline) for cross-domain queries. This is a consistent advantage with lower variance (+/-4.1% vs +/-5.6%).

### GraphRAG vs Other Strategies (Matched Configurations)

When comparing identical configurations (same collection, search, alpha, top_k):

| Comparison | GraphRAG Advantage (Recall) |
|------------|----------------------------|
| GraphRAG vs Decomposition | +8.8 to +15.1 pp |
| GraphRAG vs None | -0.3 to +1.3 pp |
| GraphRAG vs HyDE | +1.4 pp |

**Interpretation:** GraphRAG dramatically outperforms Decomposition (which fragments queries) but only marginally beats None/HyDE on recall. The real advantage is in **answer correctness**, where the knowledge graph provides conceptual relationships that improve synthesis.

### Best GraphRAG Configuration

```
Collection: contextual
Search: semantic (alpha=1.0)
Top_K: 10

Cross-Domain Correctness: 59.4%
Cross-Domain Recall: 73.3%
Single-Concept Correctness: 53.9%
```

### GraphRAG Synergy with Chunking Strategies

| Collection | Cross-Domain Recall | Cross-Domain Correctness |
|------------|--------------------|-----------------------|
| Contextual | 77.5% | 52.0% |
| Section | 74.7% | 48.3% |

**Recommendation:** GraphRAG works best with Contextual chunking. The LLM-generated context in chunks synergizes with the knowledge graph structure—context disambiguates "what this chunk is about" while the graph captures "how concepts relate."

### Why GraphRAG Improves Correctness More Than Recall

GraphRAG extracts entities and relationships from the query, traverses a pre-built knowledge graph (Neo4j with Leiden community detection), and merges graph-derived chunks with vector search results. This provides:

1. **Conceptual anchoring**: Entities like "consciousness" get connected to related concepts ("qualia", "hard problem", "Chalmers") even if those terms don't appear in the query
2. **Cross-domain bridging**: The knowledge graph encodes relationships between neuroscience and philosophy concepts discovered during extraction
3. **Reduced hallucination**: Graph-verified conceptual relationships provide grounding that pure vector similarity misses

The recall advantage is modest because vector search already finds relevant chunks. The correctness advantage is larger because the graph structure helps the generation LLM synthesize information from multiple sources correctly.

### GraphRAG Trade-offs

**Advantages:**
- Highest answer correctness (+2.4 pp over baseline)
- Best for relationship-based queries ("how do X and Y relate?")
- Provides conceptual structure for cross-domain synthesis

**Disadvantages:**
- Requires pre-built knowledge graph (Neo4j + entity extraction)
- Only tested on contextual and section collections
- Modest recall improvement over simpler strategies

**When to use GraphRAG:**
- Questions requiring conceptual integration across domains
- Corpus with rich entity relationships
- When answer correctness matters more than retrieval speed

---

## Cross-Domain Degradation Summary

| Metric | Single-Concept Mean | Cross-Domain Mean | Degradation |
|--------|---------------------|-------------------|-------------|
| Context Recall | 93.1% | 72.2% | **-22.5%** |
| Answer Relevancy | 89.1% | 65.6% | -26.4% |
| Answer Correctness | 56.2% | 47.8% | -14.9% |

Cross-domain questions are fundamentally harder: they require retrieving evidence from 4-6 different books and synthesizing concepts that span neuroscience and philosophy vocabularies.

---

## Key Takeaways (Portfolio Summary)

*This project evaluated 102 RAG configurations across a mixed neuroscience/philosophy corpus. While the small test set (15 questions) and specific domain limit generalizability, the evaluation revealed consistent patterns that informed my understanding of RAG system design.*

### 1. Foundational Choices Matter More Than Runtime Tuning

**Intuition:** How you chunk your documents has ~4x more impact on retrieval quality than runtime parameters like top_k.

Measured by Cohen's d effect sizes: chunking strategy (d=2.66) vs top_k (d=0.70) represents a 3.8x difference in effect magnitude. In absolute terms, switching from semantic chunking (loose threshold) to contextual chunking improved cross-domain recall by 23.8 percentage points—while doubling top_k from 10 to 20 only improved recall by 7.9 percentage points. This suggests that for practitioners building RAG systems, investing time in document preparation yields higher returns than optimizing query-time parameters.

*Caveat: Tested on a neuroscience + philosophy corpus; generalization to other domains needs validation.*

### 2. Cross-Domain Synthesis Is a Distinct Challenge

**Intuition:** Questions requiring information from multiple knowledge domains consistently underperform single-source questions, regardless of configuration.

Across 102 configurations, cross-domain questions showed 22.5% lower context recall than single-concept questions. No technique eliminated this gap—HyDE reduced it to 10.5%, but decomposition made it worse (30.4% degradation). This points to a fundamental challenge in RAG: retrieving coherent evidence across conceptual boundaries is harder than retrieving from a single domain, and some strategies (query decomposition) actively harm synthesis tasks.

*Caveat: Based on 15 hand-crafted questions; larger test sets needed for statistical confidence.*

### 3. Retrieval Quality ≠ Answer Quality

**Intuition:** Improving retrieval metrics doesn't guarantee better final answers—different strategies optimize different outcomes.

The configuration with best cross-domain recall (contextual + HyDE, 89.5%) ranked outside the top 5 for answer correctness. Conversely, the best answer correctness (raptor + none, 59.5%) had only 79.7% recall. GraphRAG showed a unique pattern: modest recall improvement but highest answer correctness, suggesting the knowledge graph helps synthesis more than retrieval. This highlights that RAG evaluation requires measuring both retrieval and generation quality.

*Caveat: Answer correctness is partly dependent on generation model capability, not just retrieval.*
