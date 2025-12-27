# Evaluation Benchmarks

## Standard QA Benchmarks (RAPTOR Paper)

The RAPTOR paper ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059)) evaluates on three benchmark datasets:

| Dataset | Domain | Size | Document Length | Answer Type | Primary Metric |
|---------|--------|------|-----------------|-------------|----------------|
| **QASPER** | NLP research papers | 5,049 Q / 1,585 papers | ~3K-5K tokens | Extractive, Yes/No, Free-form | Token F1 |
| **QuALITY** | Fiction stories | ~2,500 Q / 250 stories | ~5K tokens | Multiple-choice (4 options) | Accuracy |
| **NarrativeQA** | Books + movie scripts | 46,765 Q&A pairs | 60K+ tokens (books) | Free-form (synthesized) | F1 + BLEU |

### QASPER
- Questions written from paper abstracts, answered from full text
- 55.5% require multi-paragraph evidence
- Designed for information-seeking on scientific documents
- Source: [HuggingFace](https://huggingface.co/datasets/allenai/qasper)

### QuALITY
- Multiple-choice format eliminates answer ambiguity
- "Hard" subset: questions that annotators couldn't answer quickly
- Stories from Project Gutenberg (public domain)
- Source: [NYU-MLL](https://nyu-mll.github.io/quality/)

### NarrativeQA
- Two tasks: summary-only (easier) vs full-story (harder)
- Only 29.6% extractive - most answers require synthesis
- Two reference answers per question (captures variability)
- Source: [GitHub](https://github.com/google-deepmind/narrativeqa)

## RAG1-Mini Evaluation Approach

This project uses **domain-specific evaluation** rather than generic benchmarks:

1. **Custom test set**: 45 questions tailored to the neuroscience/philosophy corpus
2. **Ground truth references**: Each question has expert-written reference answers
3. **Category-based analysis**: Questions grouped by domain and difficulty

### Why Not Use Standard Benchmarks?

The RAPTOR benchmarks test different domains:
- QASPER: NLP research papers (not neuroscience/philosophy)
- QuALITY: Fiction stories (different document structure)
- NarrativeQA: Books/scripts (fiction, not technical content)

Domain-specific questions better measure retrieval quality for the actual corpus.

## F1 Score Implementation

We include two F1-based metrics for benchmark comparability:

### 1. RAGAS AnswerCorrectness (LLM-based)

```python
from ragas.metrics import AnswerCorrectness

metric_map = {
    "answer_correctness": AnswerCorrectness(),  # Weighted F1 + semantic
}
```

AnswerCorrectness combines:
- **Factual similarity** (75%): LLM decomposes into claims, classifies TP/FP/FN
- **Semantic similarity** (25%): Embedding cosine similarity

**Pros**: Understands synonyms, paraphrasing, semantic equivalence
**Cons**: Slow (~2-5s/question), costs LLM calls, non-deterministic

### 2. SQuAD-style F1 (Token-based)

```python
from src.evaluation.ragas_evaluator import compute_squad_f1

score = compute_squad_f1(prediction, reference)  # 0.0 to 1.0
```

Standard QA benchmark metric:
- Normalizes text (lowercase, remove articles/punctuation)
- Computes token overlap precision/recall
- Returns harmonic mean (F1)

**Pros**: Instant, free, deterministic, directly comparable to QASPER/NarrativeQA
**Cons**: No semantic understanding ("expands" ≠ "grows")

### When to Use Each

| Use Case | Metric |
|----------|--------|
| Production quality assessment | AnswerCorrectness |
| Benchmark paper comparison | SQuAD F1 |
| Quick iteration | SQuAD F1 |
| Final evaluation | Both |

## Running Evaluation

```bash
# Full evaluation with F1 (default metrics now include answer_correctness)
python -m src.stages.run_stage_7_evaluation

# Comprehensive grid search
python -m src.stages.run_stage_7_evaluation --comprehensive
```
