# RAG Evaluation History

Tracking all RAGAS evaluation runs with configurations and metrics for future article.

---

## Summary Table

| Run | Date | Search | Top-K | Alpha | Rerank | Gen Model | Eval Model | Relevancy | Faithfulness | Failures |
|-----|------|--------|-------|-------|--------|-----------|------------|-----------|--------------|----------|
| 1 | Dec 19 | Vector | 5 | - | No | gpt-4o-mini | gpt-4o-mini | 0.669 | 0.926 | 7/23 (30%) |
| 2 | Dec 20 | Hybrid | 10 | 0.5 | No | gpt-5-mini | claude-haiku-4.5 | 0.738 | 0.929 | 1/23 (4%) |
| 3 | Dec 20 | Hybrid | 10 | 0.5 | No | gpt-5-mini | claude-haiku-4.5 | 0.786 | 0.885 | 0/23 (0%) |
| 4 | Dec 20 | Hybrid | 10 | 0.5 | Yes | gpt-5-mini | claude-haiku-4.5 | 0.787 | 0.927 | 1/23 (4%) |

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

## Run 4: Cross-Encoder Reranking

**Date:** December 20, 2025 (10:15:07 UTC)
**File:** `data/evaluation/results/eval_20251220_101507.json`

### Configuration
- **Search Type:** Hybrid (BM25 + vector)
- **Top-K:** 10
- **Alpha:** 0.5
- **Embedding Model:** text-embedding-3-large
- **Generation Model:** openai/gpt-5-mini
- **Evaluation Model:** anthropic/claude-haiku-4.5
- **Reranking:** Yes (mxbai-rerank-large-v1, 560M params, CPU)

### Results
| Metric | Score | Change vs Run 3 |
|--------|-------|-----------------|
| Faithfulness | 0.927 | +4.7% |
| Relevancy | 0.787 | +0.1% |
| Failures | 1/23 (4%) | -1 question |

### Performance Notes
- **Reranking time:** ~2 min per question (50 docs, CPU)
- **Total runtime:** ~47 minutes for 23 questions
- Reranking changed top result in 13/23 queries (57%)

### Failed Questions
- `neuro_behave_01` - Behavioral neuroscience (faithfulness=0.0)

### Key Learning
Cross-encoder reranking improved faithfulness significantly (+4.7%) by surfacing more relevant chunks. However, CPU-based inference is extremely slow (~2 min/query). For production use, consider:
1. API-based rerankers (Cohere, Voyage, Jina) for speed
2. GPU acceleration for local inference
3. Smaller models (ms-marco-MiniLM) for faster CPU inference

---

## Improvement Opportunities

Based on research in `rag-improve-research.md`:

| Improvement | Expected Impact | Status |
|-------------|-----------------|--------|
| Cross-encoder reranking | +20-35% precision | Tested (Run 4): +4.7% faithfulness, very slow on CPU |
| Alpha tuning (0.3, 0.7) | Find optimal balance | Ready to test |
| Step-back prompting | +27% multi-hop | Implemented (step_back strategy) |
| Query decomposition | +36.7% MRR@10 | Implemented (decomposition strategy) |
| Multi-query expansion | +coverage | Implemented (multi_query strategy) |
| API-based reranking | Speed + quality | Research complete (see below) |

---

## How to Run Experiments

```bash
# Test reranking effect
python -m src.run_stage_7_evaluation --reranking      # with reranking
python -m src.run_stage_7_evaluation --no-reranking   # without

# Test alpha values
python -m src.run_stage_7_evaluation --alpha 0.3 -o data/evaluation/results/alpha_0.3.json
python -m src.run_stage_7_evaluation --alpha 0.5 -o data/evaluation/results/alpha_0.5.json
python -m src.run_stage_7_evaluation --alpha 0.7 -o data/evaluation/results/alpha_0.7.json
```

### Alpha Parameter Explained

| Alpha | Weight | Best For |
|-------|--------|----------|
| 0.0 | 100% BM25 | Exact term matching |
| 0.3 | 70% BM25 + 30% vector | Philosophy terminology |
| 0.5 | Balanced | General use (default) |
| 0.7 | 30% BM25 + 70% vector | Conceptual queries |
| 1.0 | 100% vector | Pure semantic search |

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

---

## API-Based Reranking Options (Research Dec 20)

Local cross-encoder (mxbai-rerank-large-v1) is too slow on CPU (~2 min/query). API alternatives:

| Provider | Model | Pricing | Speed | Notes |
|----------|-------|---------|-------|-------|
| **Voyage AI** | rerank-2.5 | $0.05/1M tokens | Fast | 200M free tokens, 16K context |
| **Voyage AI** | rerank-2.5-lite | $0.02/1M tokens | Faster | 8K context, 2.5x cheaper |
| **Cohere** | Rerank 3.5 | $2.00/1K searches | Fast | 1 search = 1 query + up to 100 docs |
| **Jina AI** | Reranker v2 | Token-based | Fast | 10M free tokens, 100+ languages |
| **MixedBread** | mxbai-rerank | $7.50/1K queries | Fast | Same model, cloud-hosted |

### Recommendation
For this project's scale (23 eval questions, ~50 docs each):

1. **Best value:** Voyage rerank-2.5-lite ($0.02/1M tokens)
   - Free tier covers extensive testing
   - Token-based = predictable costs

2. **Best quality:** Cohere Rerank 3.5 ($2.00/1K searches)
   - State-of-the-art accuracy
   - Simple search-based pricing

### Integration
OpenRouter does NOT currently offer reranking models through their unified API. Direct integration with Voyage/Cohere/Jina required.

**Unified library:** `pip install "rerankers[api]"` supports Cohere, Jina, MixedBread via single API.
