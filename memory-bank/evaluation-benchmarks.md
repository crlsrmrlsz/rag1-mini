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

## RAGLab Evaluation Approach

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

## Answer Correctness Metric

RAGAS `AnswerCorrectness` is the end-to-end metric for comparing generated answers to reference answers:

```python
from ragas.metrics import AnswerCorrectness

# Weighted combination:
# - Factual similarity (75%): LLM decomposes into claims, classifies TP/FP/FN
# - Semantic similarity (25%): Embedding cosine similarity
```

**Pros**: Understands synonyms, paraphrasing, semantic equivalence
**Cons**: Slow (~2-5s/question), costs LLM calls, non-deterministic

**Decision**: SQuAD-style token F1 was removed (Dec 2024) because it lacks semantic understanding ("expands" ≠ "grows") and the RAGAS metric provides better evaluation quality for RAG systems.
