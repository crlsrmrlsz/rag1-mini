# Chunking Strategy Analysis

**Date:** 2026-01-03
**Data Source:** Comprehensive RAG evaluation results across 5 chunking strategies

---

## Executive Summary

- **Contextual chunking delivers best overall performance** with highest answer correctness in both single-concept (0.591) and cross-domain (0.488) queries, making it the recommended default strategy.
- **Section (fixed-size) chunking provides best consistency** with the smallest context recall degradation (-0.166) when moving from easy to hard queries.
- **Semantic chunking at 0.75 threshold underperforms across all metrics** - lowest retrieval quality and end-to-end performance; the looser threshold creates chunks that miss semantic coherence.
- **High retrieval precision does not guarantee best answers** - Semantic 0.3 leads in single-concept precision (0.734) but ranks 4th in answer correctness (0.541).

---

## 1. Retrieval Performance Analysis

Retrieval quality is measured by two metrics:
- **Context Precision**: What fraction of retrieved chunks are relevant?
- **Context Recall**: What fraction of relevant information was retrieved?

### 1.1 Single-Concept Queries

| Chunking Strategy | Context Precision | Context Recall | Combined Avg |
|-------------------|-------------------|----------------|--------------|
| Semantic 0.3 | **0.734** | 0.933 | 0.834 |
| Contextual | 0.717 | **0.963** | **0.840** |
| Semantic 0.75 | 0.712 | 0.861 | 0.787 |
| Section | 0.691 | 0.929 | 0.810 |
| RAPTOR | 0.681 | 0.961 | 0.821 |

**Key Insights:**
- **Contextual achieves best recall** (0.963) - the LLM-generated context helps surface more relevant chunks
- **Semantic 0.3 leads in precision** (0.734) - stricter similarity threshold produces more focused chunks
- **RAPTOR's hierarchical summaries excel at recall** (0.961) but sacrifice precision (0.681)
- **Semantic 0.75's loose threshold hurts both metrics** - chunks are too broad, reducing both precision and recall

### 1.2 Cross-Domain Queries

| Chunking Strategy | Context Precision | Context Recall | Combined Avg |
|-------------------|-------------------|----------------|--------------|
| RAPTOR | **0.938** | 0.764 | **0.851** |
| Semantic 0.3 | 0.936 | 0.693 | 0.815 |
| Section | 0.927 | 0.763 | 0.845 |
| Contextual | 0.919 | **0.794** | 0.857 |
| Semantic 0.75 | 0.878 | 0.556 | 0.717 |

**Key Insights:**
- **All strategies improve precision on cross-domain** (+0.16 to +0.26) - harder queries may force more selective retrieval
- **All strategies lose recall on cross-domain** (-0.17 to -0.31) - integrating information across domains is inherently harder
- **Contextual maintains best recall** (0.794) even on cross-domain, showing its robustness
- **Semantic 0.75 collapses on cross-domain recall** (0.556) - the loosest chunking fails to capture cross-domain connections

### 1.3 Retrieval Summary

| Strategy | Best For | Weakness |
|----------|----------|----------|
| Contextual | High recall across all query types | Slightly lower precision than Semantic 0.3 |
| Semantic 0.3 | Best single-concept precision | Steeper recall drop on cross-domain (-0.24) |
| RAPTOR | Best cross-domain precision | Lower single-concept precision |
| Section | Consistent recall retention | Middle-of-the-pack on both metrics |
| Semantic 0.75 | None | Underperforms across all scenarios |

---

## 2. End-to-End Performance Analysis

End-to-end quality is measured by:
- **Answer Correctness**: Semantic similarity + factual overlap with ground truth
- **Relevancy**: How well the answer addresses the question

### 2.1 Single-Concept Queries

| Chunking Strategy | Answer Correctness | Relevancy | Faithfulness |
|-------------------|-------------------|-----------|--------------|
| Contextual | **0.591** | 0.855 | 0.939 |
| RAPTOR | 0.579 | 0.815 | **0.952** |
| Section | 0.576 | **0.891** | 0.950 |
| Semantic 0.3 | 0.541 | 0.830 | 0.902 |
| Semantic 0.75 | 0.508 | 0.830 | 0.854 |

**Key Insights:**
- **Contextual leads in answer correctness** (0.591) - the added context helps the generator produce more accurate answers
- **Section leads in relevancy** (0.891) - traditional chunking produces well-scoped chunks that align with questions
- **RAPTOR leads in faithfulness** (0.952) - hierarchical summaries reduce hallucination risk
- **Semantic 0.75 struggles with faithfulness** (0.854) - overly broad chunks may introduce irrelevant information

### 2.2 Cross-Domain Queries

| Chunking Strategy | Answer Correctness | Relevancy | Faithfulness |
|-------------------|-------------------|-----------|--------------|
| Contextual | **0.488** | **0.637** | **0.991** |
| RAPTOR | 0.484 | 0.606 | 0.981 |
| Semantic 0.3 | 0.480 | 0.591 | 0.984 |
| Section | 0.479 | 0.586 | 0.991 |
| Semantic 0.75 | 0.456 | 0.492 | 0.967 |

**Key Insights:**
- **Contextual maintains leadership in cross-domain** - wins all three metrics
- **All strategies improve faithfulness on cross-domain** - harder queries make models more cautious
- **Semantic 0.75 shows severe relevancy collapse** (0.492) - chunks fail to capture cross-domain semantics
- **Gap between strategies narrows in answer correctness** - cross-domain is hard for everyone

### 2.3 End-to-End Summary

| Strategy | Single-Concept Correctness | Cross-Domain Correctness | Overall Winner? |
|----------|---------------------------|-------------------------|-----------------|
| Contextual | 0.591 (1st) | 0.488 (1st) | **Yes** |
| RAPTOR | 0.579 (2nd) | 0.484 (2nd) | Close 2nd |
| Section | 0.576 (3rd) | 0.479 (4th) | Solid performer |
| Semantic 0.3 | 0.541 (4th) | 0.480 (3rd) | Precision doesn't convert |
| Semantic 0.75 | 0.508 (5th) | 0.456 (5th) | Avoid |

---

## 3. Single-Concept vs Cross-Domain Gap Analysis

The delta shows how much performance degrades from easy (single-concept) to hard (cross-domain) queries. **Smaller negative values indicate better robustness.**

### 3.1 Answer Correctness Degradation

| Chunking Strategy | Single | Cross | Delta | Interpretation |
|-------------------|--------|-------|-------|----------------|
| Semantic 0.75 | 0.508 | 0.456 | **-0.052** | Smallest drop but from lowest baseline |
| Semantic 0.3 | 0.541 | 0.480 | -0.062 | Better robustness than expected |
| RAPTOR | 0.579 | 0.484 | -0.095 | Hierarchical summaries help |
| Section | 0.576 | 0.479 | -0.097 | Consistent degradation |
| Contextual | 0.591 | 0.488 | -0.103 | Largest drop but maintains lead |

**Note:** Semantic 0.75's small delta is misleading - it starts from a weak baseline and remains weakest absolutely.

### 3.2 Context Recall Degradation

| Chunking Strategy | Single | Cross | Delta | Interpretation |
|-------------------|--------|-------|-------|----------------|
| Section | 0.929 | 0.763 | **-0.166** | Best recall retention |
| Contextual | 0.963 | 0.794 | -0.168 | Near-best retention |
| RAPTOR | 0.961 | 0.764 | -0.197 | Hierarchical structure helps |
| Semantic 0.3 | 0.933 | 0.693 | -0.240 | Stricter threshold hurts cross-domain |
| Semantic 0.75 | 0.861 | 0.556 | **-0.305** | Worst: chunks fail cross-domain |

**Key Finding:** Section and Contextual chunking show the best resilience in retrieving relevant information across domains.

### 3.3 Relevancy Degradation

| Chunking Strategy | Single | Cross | Delta | Interpretation |
|-------------------|--------|-------|-------|----------------|
| RAPTOR | 0.815 | 0.606 | **-0.209** | Summaries generalize well |
| Contextual | 0.855 | 0.637 | -0.218 | Context helps maintain relevancy |
| Semantic 0.3 | 0.830 | 0.591 | -0.239 | Moderate degradation |
| Section | 0.891 | 0.586 | -0.305 | Biggest relevancy drop |
| Semantic 0.75 | 0.830 | 0.492 | **-0.338** | Severe collapse |

**Key Finding:** RAPTOR's hierarchical summaries provide the best relevancy retention on harder queries.

---

## 4. Key Findings and Recommendations

### 4.1 Overall Performance Ranking

| Rank | Strategy | Strengths | Weaknesses |
|------|----------|-----------|------------|
| 1 | **Contextual** | Best answer correctness, best cross-domain recall | Largest absolute correctness drop |
| 2 | RAPTOR | Best faithfulness, good cross-domain precision | Lower single-concept precision |
| 3 | Section | Best relevancy, most consistent recall | Lower correctness than Contextual |
| 4 | Semantic 0.3 | Best single-concept precision | Poor end-to-end conversion |
| 5 | Semantic 0.75 | None significant | Underperforms everywhere |

### 4.2 Strategy Selection Guide

| Use Case | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| **General production** | Contextual | Best overall answer quality |
| **Low-hallucination required** | RAPTOR | Highest faithfulness scores |
| **Consistent behavior needed** | Section | Most stable across query types |
| **Single-domain knowledge base** | Semantic 0.3 | Good precision when cross-domain not needed |
| **Avoid** | Semantic 0.75 | No use case where it excels |

### 4.3 Best for Retrieval vs Best for End-to-End

| Metric Type | Best Strategy | Value |
|-------------|---------------|-------|
| Best single-concept precision | Semantic 0.3 | 0.734 |
| Best single-concept recall | Contextual | 0.963 |
| Best cross-domain precision | RAPTOR | 0.938 |
| Best cross-domain recall | Contextual | 0.794 |
| Best single-concept correctness | **Contextual** | **0.591** |
| Best cross-domain correctness | **Contextual** | **0.488** |

**Important Finding:** High retrieval precision (Semantic 0.3) does not translate to best answers. **Recall matters more** - Contextual's high recall (0.963) correlates with best answer correctness despite having lower precision than Semantic 0.3.

### 4.4 Trade-offs Summary

```
                    Precision ←→ Recall
                         ↑
    Semantic 0.3 ----→ Sweet spot for focused retrieval
                         |
    Contextual  ----→ Best overall (favors recall)
                         |
    RAPTOR      ----→ Good for cross-domain, high faithfulness
                         |
    Section     ----→ Consistent baseline, good relevancy
                         |
    Semantic 0.75 --→ Avoid (worst on most metrics)
                         ↓
```

---

## 5. Statistical Confidence

| Strategy | N (Single) | N (Cross) | Data Quality |
|----------|------------|-----------|--------------|
| Contextual | 24 | 24 | High |
| Section | 24 | 24 | High |
| RAPTOR | 18 | 18 | Moderate |
| Semantic 0.3 | 18 | 18 | Moderate |
| Semantic 0.75 | 18 | 18 | Moderate |

Standard deviations for answer correctness range from 0.04-0.08, indicating reasonably stable measurements but some variance across questions.

---

## 6. Recommendations

1. **Default to Contextual chunking** for new RAG pipelines - it provides the best end-to-end answer quality.

2. **Use RAPTOR when hallucination is critical** - its 0.95+ faithfulness scores are valuable for high-stakes applications.

3. **Consider Section chunking as a baseline** - it's the most predictable and shows good relevancy scores.

4. **Avoid Semantic 0.75** - the loose threshold creates poor chunk boundaries that harm both retrieval and generation.

5. **If using Semantic chunking, prefer 0.3 threshold** - stricter similarity creates more coherent chunks, though end-to-end performance still lags Contextual.

6. **Recall is more important than precision for answer quality** - prioritize chunking strategies that maximize recall when answer correctness is the goal.
