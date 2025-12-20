# RAG Evaluation History

Tracking all RAGAS evaluation runs with configurations and metrics for future article.

---

## Summary Table

| Run | Date | Search | Top-K | Alpha | Gen Model | Eval Model | Relevancy | Faithfulness | Failures |
|-----|------|--------|-------|-------|-----------|------------|-----------|--------------|----------|
| 1 | Dec 19 | Vector | 5 | - | gpt-4o-mini | gpt-4o-mini | 0.669 | 0.926 | 7/23 (30%) |
| 2 | Dec 20 | Hybrid | 10 | 0.5 | gpt-5-mini | claude-haiku-4.5 | 0.738 | 0.929 | 1/23 (4%) |
| 3 | Dec 20 | Hybrid | 10 | 0.5 | gpt-5-mini | claude-haiku-4.5 | 0.786 | 0.885 | 0/23 (0%) |

---

## Run 1: Vector Search Baseline

**Date:** December 19, 2025
**File:** `data/evaluation/results/eval_20251219_134548.json`

### Configuration
- **Search Type:** Pure vector similarity (`query_similar`)
- **Top-K:** 5
- **Alpha:** N/A
- **Embedding Model:** text-embedding-3-large
- **Generation Model:** openai/gpt-4o-mini
- **Evaluation Model:** openai/gpt-4o-mini
- **Reranking:** No

### Results
| Metric | Score |
|--------|-------|
| Faithfulness | 0.926 |
| Relevancy | 0.669 |
| Failures | 7/23 (30%) |

### Category Breakdown
| Category | Relevancy | Faithfulness |
|----------|-----------|--------------|
| Neuroscience (8) | 0.72 | 0.96 |
| Philosophy (11) | 0.56 | 0.89 |
| Cross-domain (4) | 0.97 | 0.98 |

### Failed Questions
- `neuro_behave_01` - Behavioral neuroscience query
- `neuro_eagleman_01` - Eagleman concepts
- `phil_seneca_01` - Stoic philosophy
- `phil_marcus_01` - Marcus Aurelius
- `phil_tao_01` - Tao paradox
- `phil_confucius_01` - Confucian concepts
- `phil_gracian_01` - Gracian's "engaño" terminology

### Key Learning
Pure vector search struggles with philosophical terminology. Exact term matching needed for concepts like "puppet metaphor", "Stoic", "Tao paradox". Philosophy category significantly underperforms (0.56 vs 0.72 for neuro).

---

## Run 2: Hybrid Search Introduction

**Date:** December 20, 2025 (05:12:45 UTC)
**File:** `data/evaluation/results/eval_20251220_051245.json`

### Configuration
- **Search Type:** Hybrid (BM25 + vector)
- **Top-K:** 10
- **Alpha:** 0.5 (balanced)
- **Embedding Model:** text-embedding-3-large
- **Generation Model:** openai/gpt-5-mini
- **Evaluation Model:** anthropic/claude-haiku-4.5
- **Reranking:** No

### Results
| Metric | Score | Change vs Run 1 |
|--------|-------|-----------------|
| Faithfulness | 0.929 | +0.3% |
| Relevancy | 0.738 | +10.3% |
| Context Precision | 0.65 | (new metric) |
| Failures | 1/23 (4%) | -85.7% |

### Category Breakdown
| Category | Relevancy | Faithfulness |
|----------|-----------|--------------|
| Neuroscience (8) | 0.77 | 0.92 |
| Philosophy (11) | 0.68 | 0.94 |
| Cross-domain (4) | 0.77 | 0.96 |

### Failed Questions
- `phil_gracian_01` - Still fails (0.0 relevancy)

### Key Learning
Hybrid search with BM25 recovers most philosophical terminology failures. Doubling top-k (5→10) provides more candidates for answer synthesis. Only Gracian's specialized Spanish terminology ("engaño") still fails.

---

## Run 3: Optimized Hybrid (Current Best)

**Date:** December 20, 2025 (05:35:56 UTC)
**File:** `data/evaluation/results/eval_20251220_053556.json`
**Commit:** c1e972b

### Configuration
- **Search Type:** Hybrid (BM25 + vector)
- **Top-K:** 10
- **Alpha:** 0.5
- **Embedding Model:** text-embedding-3-large
- **Generation Model:** openai/gpt-5-mini
- **Evaluation Model:** anthropic/claude-haiku-4.5
- **Reranking:** No
- **Query Preprocessing:** None

### Results
| Metric | Score | Change vs Run 1 | Change vs Run 2 |
|--------|-------|-----------------|-----------------|
| Faithfulness | 0.885 | -4.4% | -4.7% |
| Relevancy | 0.786 | +17.5% | +6.5% |
| Context Precision | 0.73 | (new) | +12.3% |
| Failures | 0/23 (0%) | -100% | -100% |

### Category Breakdown
| Category | Relevancy | Faithfulness |
|----------|-----------|--------------|
| Neuroscience (8) | 0.80 | 0.84 |
| Philosophy (11) | 0.75 | 0.93 |
| Cross-domain (4) | 0.78 | 0.88 |

### All Questions Passed
No failures - including previously problematic:
- Puppet metaphor retrieval ✓
- Engaño/deceit terminology ✓
- Tao paradox questions ✓

### Key Learning
Consistent hybrid configuration achieves zero failures. Philosophy category now competitive (0.75 vs 0.80 neuro). Trade-off: slightly lower faithfulness (0.885 vs 0.926) suggests more aggressive retrieval may include tangential content.

---

## Improvement Opportunities

Based on research in `rag-improve-research.md`:

| Improvement | Expected Impact | Status |
|-------------|-----------------|--------|
| Cross-encoder reranking | +20-35% precision | Scaffolded, not tested |
| Alpha tuning (0.3, 0.7) | Find optimal balance | Pending |
| Step-back prompting | +27% multi-hop | Pending |
| Query decomposition | +36.7% MRR@10 | Pending |

---

## Technical Notes

### API Issues Encountered
- RAGAS `llm_factory/embedding_factory` API incompatible (commit abb52a8)
- Reverted to `LangchainLLMWrapper` (deprecated but functional)
- Error: "InstructorLLM has no attribute agenerate_prompt"

### Model Selection Rationale
- **Generation (gpt-5-mini):** Cost-effective, good reasoning
- **Evaluation (claude-haiku-4.5):** RAGAS research shows Anthropic models are stable judges
- **Embedding (text-embedding-3-large):** High-quality, 3072 dimensions
