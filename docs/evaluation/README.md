# Evaluation Framework

RAGLab uses [RAGAS](https://docs.ragas.io/) (Retrieval-Augmented Generation Assessment) for systematic evaluation of RAG strategy combinations.

## Metrics

| Metric | What It Measures | Range |
|--------|------------------|-------|
| **Faithfulness** | Are claims in the answer supported by retrieved context? | 0-1 |
| **Answer Relevancy** | Does the answer address the question? | 0-1 |
| **Context Precision** | Are retrieved chunks actually relevant? | 0-1 |
| **Answer Correctness** | F1 overlap with reference answer | 0-1 |

### Metric Details

**Faithfulness** (most important for avoiding hallucination):
- Extracts claims from generated answer
- Checks if each claim can be inferred from context
- Low score = hallucination or unsupported statements

**Answer Relevancy**:
- Generates synthetic questions from answer
- Compares to original question
- Low score = off-topic or incomplete answer

**Context Precision**:
- Uses reference answer to judge context quality
- High-ranked irrelevant chunks hurt score
- Low score = retrieval returning wrong content

**Answer Correctness**:
- Token-level F1 with reference answer
- Measures factual accuracy
- Requires human-written reference answers

## Test Dataset

- **45 questions** covering neuroscience and philosophy
- **Human-written reference answers** for each
- Categories: factual, conceptual, comparative, synthesis

Example:
```json
{
  "question": "How does chronic stress affect the hippocampus?",
  "ground_truth": "Chronic stress causes elevated cortisol levels that damage hippocampal neurons, reduce neurogenesis, and impair memory consolidation. Sapolsky's research shows prolonged stress can shrink hippocampal volume."
}
```

## Evaluation Grid

```
                    CHUNKING STRATEGIES
              section    contextual    raptor
            ┌──────────┬────────────┬──────────┐
    none    │          │            │          │
            ├──────────┼────────────┼──────────┤
    hyde    │   (12 combinations total)         │
PREPROCESSING ├──────────┼────────────┼──────────┤
    decomp  │          │            │          │
            ├──────────┼────────────┼──────────┤
    graphrag│          │            │          │
            └──────────┴────────────┴──────────┘
```

Plus alpha parameter (BM25/vector balance): 0.0, 0.3, 0.5, 0.7, 1.0

**Total configurations**: 3 chunking × 4 preprocessing × 5 alphas = **60 evaluations**

## Running Evaluation

```bash
# Single configuration
python -m src.stages.run_stage_7_evaluation \
  --collection RAG_section_embed3large_v1 \
  --preprocessing hyde \
  --alpha 0.7

# Grid search (comprehensive)
python -m src.stages.run_stage_7_evaluation --comprehensive
```

The comprehensive mode uses a curated 10-question subset for faster iteration.

## Output

Results are appended to `memory-bank/evaluation-history.md`:

```markdown
## Run: 2025-12-28 14:30

**Config:** section + hyde, alpha=0.7
**Questions:** 10 (comprehensive subset)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.82 |
| Answer Relevancy | 0.78 |
| Context Precision | 0.71 |
| Answer Correctness | 0.65 |
```

## Key Files

| File | Purpose |
|------|---------|
| `src/evaluation/ragas_evaluator.py` | RAGAS wrapper |
| `src/evaluation/dataset.py` | Question loading |
| `src/stages/run_stage_7_evaluation.py` | CLI runner |
| `data/evaluation/questions.json` | Full 45-question set |
| `data/evaluation/comprehensive_questions.json` | 10-question subset |
| `memory-bank/evaluation-history.md` | Run history |

## Results

See [Results](results.md) for detailed metrics across all configurations.
