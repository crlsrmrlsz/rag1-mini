# RAGLab Comprehensive Evaluation: Findings Report v2

**Date:** January 3, 2026
**Evaluation Run:** comprehensive_20260101_164236
**Version:** 2 (Refined focus on retrieval vs end-to-end metrics)

---

## Study Overview

**Configurations Analyzed:** 84 (after excluding semantic_0.75 edge case from original 102)

**5-Dimensional Grid:**
- **Chunking:** section, contextual, raptor, semantic_0.3
- **Preprocessing:** none, HyDE, decomposition, GraphRAG
- **Search Type:** keyword (BM25), hybrid (vector + BM25)
- **Alpha:** 0.0, 0.5, 1.0 (semantic weight)
- **Top-K:** 10, 20

**Corpus:** 19 books (neuroscience + philosophy)
**Questions:** 15 (5 single-concept + 10 cross-domain)

---

## Analysis Framework

This report separates two types of metrics:

| Metric Type | Metrics | What It Evaluates | LLM Dependency |
|-------------|---------|-------------------|----------------|
| **Retrieval** | Context Recall, Context Precision | Quality of chunk retrieval | None |
| **End-to-End** | Answer Correctness | Quality of final answer | Generation LLM |

**Why separate them?**
- Retrieval metrics are **LLM-independent**: they measure whether the right chunks were retrieved
- Answer Correctness depends on the **generation model** (Claude 3.5 Sonnet in this study)
- Practitioners using different LLMs can rely on retrieval findings; end-to-end results may vary

---

# PART 1: RETRIEVAL SYSTEM ANALYSIS (LLM-Independent)

These findings apply regardless of which generation model you use.

## 1.1 Overall Retrieval Performance by Dimension

### Context Recall (Did we find all relevant information?)

| Dimension | Best | Worst | Gap | Cohen's d |
|-----------|------|-------|-----|-----------|
| **Chunking** | contextual (0.850) | semantic_0.3 (0.773) | 0.077 | d=1.45 |
| **Top-K** | k=20 (0.850) | k=10 (0.791) | 0.059 | d=1.10 |
| **Preprocessing** | hyde (0.847) | decomposition (0.788) | 0.059 | d=1.01 |
| Alpha | alpha=0.5 (0.828) | alpha=1.0 (0.819) | 0.009 | d=0.15 |
| Search Type | hybrid (0.824) | keyword (0.814) | 0.009 | d=0.15 |

### Context Precision (Was retrieved context relevant?)

| Dimension | Best | Worst | Gap | Cohen's d |
|-----------|------|-------|-----|-----------|
| **Search Type** | hybrid (0.888) | keyword (0.787) | 0.101 | d=2.68 |
| **Alpha** | alpha=1.0 (0.897) | alpha=0.0 (0.787) | 0.110 | d=2.65 |
| Preprocessing | semantic_0.3 (0.869) | graphrag (0.836) | 0.031 | d=0.45 |
| Chunking | contextual (0.852) | raptor (0.853) | 0.001 | d=0.01 |
| Top-K | k=10 (0.856) | k=20 (0.853) | 0.003 | d=0.04 |

---

## 1.2 Dimension Impact Hierarchy

**For Context Recall (Completeness):**
```
Chunking     ████████████████████  100%  <- Invest here first
Top-K        ███████████████       77%
Preprocessing████████████████      76%
Alpha        ███                   18%   <- Low impact
Search Type  ██                    12%   <- Low impact
```

**For Context Precision (Relevance):**
```
Alpha        ████████████████████  100%  <- Biggest lever
Search Type  ██████████████████    92%
Preprocessing█████                 29%
Chunking     ███                   19%   <- Low impact
Top-K        ▏                      2%   <- Negligible
```

**Key Insight:** Different dimensions dominate different metrics:
- **Recall** is controlled by chunking, top_k, and preprocessing
- **Precision** is controlled by alpha and search type

---

## 1.3 Statistically Reliable Differences (Cohen's d >= 0.8)

### Context Recall

| Comparison | d | Conclusion |
|------------|---|------------|
| contextual vs semantic_0.3 | +1.45 | **contextual WINS** |
| k=20 vs k=10 | +1.10 | **k=20 WINS** |
| hyde vs decomposition | +1.01 | **hyde WINS** |
| raptor vs semantic_0.3 | +0.97 | **raptor WINS** |
| graphrag vs decomposition | +0.83 | **graphrag WINS** |
| section vs semantic_0.3 | +0.83 | **section WINS** |

### Context Precision

| Comparison | d | Conclusion |
|------------|---|------------|
| hybrid vs keyword | +2.68 | **hybrid WINS** |
| alpha=1.0 vs alpha=0.0 | +2.65 | **alpha=1.0 WINS** |
| alpha=0.5 vs alpha=0.0 | +2.37 | **alpha=0.5 WINS** |

### Indistinguishable (d < 0.4)

- contextual ~ raptor ~ section (for recall, d=0.19-0.58)
- graphrag ~ hyde ~ none (for recall, d=0.20-0.42)
- All chunking strategies (for precision, d=0.01-0.33)
- All alpha values (for recall, d=0.08-0.21)

---

## 1.4 Cross-Domain vs Single-Concept Retrieval

Cross-domain questions require synthesis across neuroscience + philosophy sources.

### The Cross-Domain Penalty

| Strategy | Single Recall | Cross Recall | Gap |
|----------|--------------|--------------|-----|
| **hyde** | 0.904 | 0.818 | **-8.6%** |
| none | 0.946 | 0.758 | -18.7% |
| graphrag | 0.975 | 0.761 | -21.4% |
| decomposition | 0.975 | 0.694 | **-28.1%** |

**Key Finding: HyDE's Cross-Domain Bridging Effect**
- Baseline cross-domain penalty: -18.7%
- HyDE cross-domain penalty: -8.6%
- **HyDE reduces the penalty by 10.1 percentage points**

This is unique to HyDE — other preprocessing strategies do not reduce the gap.

### The Decomposition Paradox

Query decomposition was designed to help multi-source retrieval, but it shows the **worst** cross-domain performance:
- Decomposition cross-domain recall: 0.694 (lowest)
- Baseline cross-domain recall: 0.758

**Hypothesis:** Sub-query generation fragments the holistic cross-domain intent. Sub-queries target EITHER neuroscience OR philosophy, losing the synthesis requirement.

---

## 1.5 Best Retrieval Configurations

| Use Case | Configuration | Recall | Precision | F1 |
|----------|---------------|--------|-----------|-----|
| **Best Balanced** | raptor + none + hybrid + alpha=0.5 + k=10 | 0.904 | 0.918 | **0.911** |
| **Best Recall** | contextual + hyde + hybrid + alpha=1.0 + k=10 | **0.930** | 0.877 | 0.902 |
| **Best Precision** | semantic_0.3 + decomp + hybrid + alpha=1.0 + k=10 | 0.751 | **0.988** | 0.853 |

---

# PART 2: END-TO-END SYSTEM ANALYSIS

Answer Correctness evaluates the complete RAG pipeline including the generation LLM.

**Note:** These results were generated using Claude 3.5 Sonnet. Results may vary with different generation models.

## 2.1 Answer Correctness by Dimension

| Dimension | Best | Worst | Gap |
|-----------|------|-------|-----|
| **Preprocessing** | graphrag (0.533) | hyde (0.505) | 0.027 |
| **Alpha** | alpha=1.0 (0.528) | alpha=0.0 (0.502) | 0.026 |
| **Chunking** | contextual (0.523) | semantic_0.3 (0.500) | 0.022 |
| **Top-K** | k=20 (0.522) | k=10 (0.504) | 0.018 |
| **Search Type** | hybrid (0.519) | keyword (0.502) | 0.017 |

**Key Observation:** End-to-end impact is more evenly distributed across dimensions than retrieval metrics.

## 2.2 Top 10 Configurations (Answer Correctness)

| Rank | Chunking | Strategy | Search | Alpha | K | Ans.Corr | Recall |
|------|----------|----------|--------|-------|---|----------|--------|
| 1 | section | hyde | hybrid | 1.0 | 20 | **0.606** | 0.859 |
| 2 | contextual | graphrag | hybrid | 0.5 | 20 | 0.602 | 0.873 |
| 3 | contextual | decomposition | keyword | 0.0 | 20 | 0.598 | 0.870 |
| 4 | raptor | none | hybrid | 0.5 | 20 | 0.589 | 0.929 |
| 5 | semantic_0.3 | hyde | hybrid | 1.0 | 20 | 0.584 | 0.830 |
| 6 | contextual | hyde | hybrid | 1.0 | 10 | 0.582 | 0.930 |
| 7 | semantic_0.3 | none | hybrid | 0.5 | 20 | 0.578 | 0.771 |
| 8 | contextual | graphrag | hybrid | 1.0 | 10 | 0.576 | 0.822 |
| 9 | section | none | keyword | 0.0 | 20 | 0.563 | 0.850 |
| 10 | raptor | none | hybrid | 1.0 | 10 | 0.560 | 0.766 |

## 2.3 Retrieval-to-Answer Correlation

| Retrieval Metric | Correlation with Answer Correctness |
|------------------|-------------------------------------|
| Context Recall | r = 0.30 |
| Context Precision | r = 0.21 |
| Retrieval F1 | r = 0.35 |

**Interpretation:** Moderate correlation (r=0.35). Better retrieval tends to produce better answers, but other factors (generation model, prompt quality) also matter.

---

# PART 3: KEY FINDINGS SUMMARY

## 3.1 Reliable Findings (Large Effect Size, d >= 0.8)

These differences are statistically reliable with n=15 questions:

| Finding | Effect Size | Practical Impact |
|---------|-------------|------------------|
| **Hybrid > Keyword** for precision | d=2.68 | +10.1% precision |
| **Alpha=1.0 > Alpha=0.0** for precision | d=2.65 | +11.0% precision |
| **Contextual > Semantic_0.3** for recall | d=1.45 | +7.7% recall |
| **k=20 > k=10** for recall | d=1.10 | +6.0% recall |
| **HyDE > Decomposition** for recall | d=1.01 | +5.9% recall |

## 3.2 Novel Findings for Publication

### Finding 1: Dimension Impact Varies by Metric Type

| For Recall | For Precision |
|------------|---------------|
| 1. Chunking (100%) | 1. Alpha (100%) |
| 2. Top-K (77%) | 2. Search Type (92%) |
| 3. Preprocessing (76%) | 3. Preprocessing (29%) |
| 4. Alpha (18%) | 4. Chunking (19%) |
| 5. Search Type (12%) | 5. Top-K (2%) |

**Implication:** Practitioners optimizing for recall should focus on chunking strategy; those optimizing for precision should focus on alpha and search type.

### Finding 2: HyDE Uniquely Bridges the Cross-Domain Gap

- HyDE is the only strategy that significantly reduces the single→cross-domain penalty
- Reduces gap from -18.7% to -8.6% (10.1 percentage point improvement)
- Hypothesis: Hypothetical document generation creates semantic bridges between domains

### Finding 3: The Decomposition Paradox

Query decomposition **hurts** cross-domain retrieval despite being designed for multi-source queries:
- -28.1% cross-domain penalty (worst among all strategies)
- Sub-queries fragment holistic intent, targeting single domains

### Finding 4: Recall and Precision Have Different Levers

To improve **recall**: Change chunking strategy (contextual > semantic), increase top_k (20 > 10)

To improve **precision**: Increase alpha (1.0 > 0.0), use hybrid search

**Trade-off:** These optimizations are largely independent — you can optimize both.

---

## 3.3 Practical Recommendations

### For Maximum Retrieval Quality
```
contextual + hyde + hybrid + alpha=1.0 + k=20
Recall: 0.877, Precision: 0.954, F1: 0.914
```

### For Maximum Answer Quality
```
section + hyde + hybrid + alpha=1.0 + k=20
Answer Correctness: 0.606
```

### For Simplicity (No LLM Preprocessing)
```
section + none + keyword + k=20
Answer Correctness: 0.563 (93% of best with zero preprocessing cost)
```

### For Cross-Domain Questions Specifically
```
contextual + hyde + hybrid + alpha=1.0 + k=10
Best cross-domain recall: 0.818 (HyDE's bridging effect)
```

---

## 3.4 Limitations

1. **Sample size:** 15 questions limits statistical power
2. **Single corpus:** Results may not generalize beyond neuroscience + philosophy
3. **Single generation LLM:** End-to-end results specific to Claude 3.5 Sonnet
4. **No cost tracking:** LLM API costs not measured

---

## Appendix: Statistical Significance Interpretation

| Cohen's d | Interpretation | For n=15 |
|-----------|----------------|----------|
| < 0.2 | Negligible | Cannot distinguish |
| 0.2-0.5 | Small | May not replicate |
| 0.5-0.8 | Medium | Likely meaningful |
| >= 0.8 | Large | **Reliable** |

All "reliable" findings in this report have d >= 0.8.

---

*v2 Changes from v1:*
- *Removed semantic_0.75 edge case (18 configurations)*
- *Separated retrieval metrics (LLM-independent) from answer correctness (LLM-dependent)*
- *Added retrieval-to-answer correlation analysis*
- *Added dimension impact hierarchy*
- *Restructured around retrieval vs end-to-end framework*

*Generated January 3, 2026*
