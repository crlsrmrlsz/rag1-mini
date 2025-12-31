# Evaluation Framework

RAGLab uses [RAGAS](https://docs.ragas.io/) (Retrieval-Augmented Generation Assessment) for systematic evaluation of RAG strategy combinations.

## Metrics

| Metric | Category | What It Measures | Requires Reference |
|--------|----------|------------------|-------------------|
| **Faithfulness** | Generation | Are claims grounded in retrieved context? | No |
| **Relevancy** | Generation | Does the answer address the question? | No |
| **Context Precision** | Retrieval | Are retrieved chunks actually relevant? | No |
| **Context Recall** | Retrieval | Did retrieval capture all needed info? | Yes |
| **Answer Correctness** | End-to-end | Is the answer factually correct? | Yes |

### Metric Details

- **Faithfulness**: Extracts claims from answer, verifies each is supported by context. Primary metric for hallucination detection.
- **Relevancy**: Generates synthetic questions from answer, compares to original. Low = off-topic answer.
- **Context Precision**: Measures if top-ranked chunks are relevant. Low = retrieval returning wrong content.
- **Context Recall**: Compares retrieved contexts to reference answer. Measures completeness of retrieval.
- **Answer Correctness**: Weighted combination of factual similarity (75%) and semantic similarity (25%).

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

## Evaluation Grid (4D)

```
Collections × Alphas × Top-K × Strategies
    │           │        │        │
    │           │        │        └── [none, hyde, decomposition, graphrag]
    │           │        └── [10, 20] chunks retrieved
    │           └── [0.0, 0.3, 0.5, 0.7, 1.0] (BM25 ↔ vector balance)
    └── [section, contextual, raptor] (chunking strategies)
```

**Typical grid**: 3 collections × 5 alphas × 2 top_k × 4 strategies = **~120 combinations**

Note: graphrag only compatible with section/contextual collections (requires matching chunk IDs).

## Running Evaluation

```bash
# Single configuration (full 45 questions)
python -m src.stages.run_stage_7_evaluation \
  --collection RAG_section_embed3large_v1 \
  --preprocessing hyde \
  --alpha 0.7 \
  --top-k 15

# Grid search (15-question curated subset)
python -m src.stages.run_stage_7_evaluation --comprehensive

# Retry failed combinations from previous run
python -m src.stages.run_stage_7_evaluation --retry-failed comprehensive_20251231_120000
```

Comprehensive mode uses 15 curated questions (5 single-concept + 10 cross-domain) for faster grid search.

## Output

### Standard Mode
- **JSON report**: `data/evaluation/ragas_results/eval_{timestamp}.json`
- **Trace file**: `data/evaluation/traces/trace_{run_id}.json` (enables metric recalculation)

### Comprehensive Mode
- **Leaderboard JSON**: `data/evaluation/ragas_results/comprehensive_{timestamp}.json`
- **Checkpoint**: `comprehensive_checkpoint_{timestamp}.json` (crash recovery)
- **Failed runs**: `failed_combinations_{timestamp}.json` (for retry)

## Key Files

| File | Purpose |
|------|---------|
| `src/evaluation/ragas_evaluator.py` | RAGAS metrics + strategy-aware retrieval |
| `src/evaluation/schemas.py` | QuestionTrace, EvaluationTrace, FailedCombination |
| `src/stages/run_stage_7_evaluation.py` | CLI runner + comprehensive grid search |
| `src/evaluation/test_questions.json` | Full 45-question test set |
| `src/evaluation/comprehensive_questions.json` | 15-question curated subset |
| `data/evaluation/traces/` | Per-run trace files (JSON) |

## Architecture

See [evaluation-workflow.md](../../memory-bank/evaluation-workflow.md) for strategy diagrams and design decisions.
