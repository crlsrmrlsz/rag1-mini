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

To enable benchmark-comparable metrics, we include **RAGAS AnswerCorrectness**:

```python
# src/evaluation/ragas_evaluator.py
from ragas.metrics import AnswerCorrectness

metric_map = {
    # ... other metrics ...
    "answer_correctness": AnswerCorrectness(),  # Includes F1 component
}
```

AnswerCorrectness combines:
- **Factual similarity**: Semantic overlap with reference
- **Answer similarity**: Token-level F1 score

This provides comparable metrics to QASPER and NarrativeQA evaluations while using domain-specific questions.

## Running Evaluation

```bash
# Full evaluation with F1 (default metrics now include answer_correctness)
python -m src.stages.run_stage_7_evaluation

# Comprehensive grid search
python -m src.stages.run_stage_7_evaluation --comprehensive
```
