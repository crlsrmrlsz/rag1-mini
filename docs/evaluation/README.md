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

## Evaluation Grid (5D)

```
Collections × Search Types × Alphas × Strategies × Top-K
    │             │           │          │           │
    │             │           │          │           └── [10, 20] chunks retrieved
    │             │           │          └── [none, hyde, decomposition, graphrag]
    │             │           └── [0.5, 1.0] (hybrid only; N/A for keyword)
    │             └── [keyword, hybrid]
    └── [section, contextual, semantic, raptor]
```

**Dimensions:**
- **Collections**: Chunking strategies (section, contextual, semantic, raptor)
- **Search Types**: `keyword` (BM25 only) or `hybrid` (vector + BM25)
- **Alphas**: For hybrid search, balance between vector (1.0) and keyword (0.5). Ignored for keyword search.
- **Strategies**: Query preprocessing (none, hyde, decomposition, graphrag)
- **Top-K**: Number of chunks to retrieve (10, 20)

**Total**: ~102 valid combinations (51 base × 2 top_k values)

Note: graphrag only compatible with section/contextual collections (requires matching chunk IDs).

## Running Evaluation

```bash
# Single configuration with hybrid search (full 45 questions)
python -m src.stages.run_stage_7_evaluation \
  --collection RAG_section_embed3large_v1 \
  --search-type hybrid \
  --preprocessing hyde \
  --alpha 0.7 \
  --top-k 15

# Single configuration with keyword (BM25) search
python -m src.stages.run_stage_7_evaluation \
  --collection RAG_section_embed3large_v1 \
  --search-type keyword \
  --preprocessing decomposition

# Grid search (15-question curated subset)
python -m src.stages.run_stage_7_evaluation --comprehensive

# Retry failed combinations from previous run
python -m src.stages.run_stage_7_evaluation --retry-failed comprehensive_20251231_120000
```

**CLI Arguments:**
| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--search-type`, `-s` | keyword, hybrid | hybrid | Weaviate query method |
| `--preprocessing`, `-p` | none, hyde, decomposition, graphrag | none | Query transformation |
| `--alpha`, `-a` | 0.0-1.0 | 0.5 | Hybrid balance (ignored for keyword) |
| `--top-k`, `-k` | int | 10 | Chunks to retrieve |
| `--collection` | string | auto | Weaviate collection name |
| `--comprehensive` | flag | - | Run 5D grid search |

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
