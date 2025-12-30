"""Stage 7: RAGAS Evaluation for RAGLab.

Runs RAGAS evaluation on test questions to measure RAG pipeline quality.
Evaluates retrieval (context precision/recall) and generation (faithfulness/relevancy).

Purpose:
    - Measure RAG pipeline quality with standardized RAGAS metrics
    - Compare different retrieval strategies (collections, alpha, reranking)
    - Auto-log results to evaluation-history.md for A/B testing
    - Track configurations in tracking.json for reproducibility

Usage Examples:
    # Run with defaults (hybrid search, alpha=0.5, reranking enabled)
    python -m src.stages.run_stage_7_evaluation

    # Test a specific collection (e.g., contextual embeddings)
    python -m src.stages.run_stage_7_evaluation --collection RAG_contextual_embed3large_v1

    # Alpha tuning experiments
    python -m src.stages.run_stage_7_evaluation --alpha 0.3  # Keyword-heavy (philosophy)
    python -m src.stages.run_stage_7_evaluation --alpha 0.7  # Vector-heavy (conceptual)

    # Disable reranking for speed comparison
    python -m src.stages.run_stage_7_evaluation --no-reranking

    # Run on subset of questions
    python -m src.stages.run_stage_7_evaluation --questions 5

    # Use different models
    python -m src.stages.run_stage_7_evaluation --generation-model openai/gpt-4o
    python -m src.stages.run_stage_7_evaluation --evaluation-model anthropic/claude-3-5-sonnet

    # Custom output path
    python -m src.stages.run_stage_7_evaluation -o data/evaluation/results/alpha_0.3.json

    # COMPREHENSIVE MODE: Test all combinations (collections x alphas x strategies)
    python -m src.stages.run_stage_7_evaluation --comprehensive
    # Uses curated 10-question subset, tests all collections, alphas (0.0-1.0), strategies
    # Generates leaderboard report with metric breakdowns

Arguments:
    -n, --questions N         Limit to first N questions
    -m, --metrics METRICS     Metrics to compute (default: faithfulness relevancy context_precision)
    -k, --top-k K             Chunks to retrieve per question (default: 10)
    -a, --alpha ALPHA         Hybrid search balance: 0.0=keyword, 0.5=balanced, 1.0=vector
    --collection NAME         Weaviate collection to evaluate (default: from config)
    --reranking/--no-reranking  Enable/disable cross-encoder reranking (default: enabled)
    --generation-model MODEL  Answer generation model (default: openai/gpt-5-mini)
    --evaluation-model MODEL  RAGAS judge model (default: anthropic/claude-haiku-4.5)
    -o, --output PATH         Output JSON file path (default: results/eval_TIMESTAMP.json)
    --no-log                  Skip auto-logging to evaluation-history.md
    --comprehensive           Run grid search across all collections, alphas, strategies

Output:
    1. JSON report: data/evaluation/results/eval_TIMESTAMP.json
    2. Markdown log: memory-bank/evaluation-history.md (auto-appended)
    3. Config tracking: data/evaluation/tracking.json (auto-updated)

Prerequisites:
    - Weaviate must be running (docker compose up -d)
    - Stage 6 must have been run to populate the collection
    - OpenRouter API key must be set in .env

See Also:
    - memory-bank/evaluation-history.md - Historical run comparisons
    - memory-bank/rag-improvement-plan.md - Improvement roadmap
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import (
    DEFAULT_TOP_K,
    EVAL_GENERATION_MODEL,
    EVAL_EVALUATION_MODEL,
    EVAL_TEST_QUESTIONS_FILE,
    EVAL_RESULTS_DIR,
    PROJECT_ROOT,
    MAX_CHUNK_TOKENS,
    OVERLAP_SENTENCES,
    EMBEDDING_MODEL,
    get_collection_name,
)
from src.evaluation import run_evaluation
from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg, OverwriteMode

# Comprehensive evaluation imports (lazy-loaded in function)
COMPREHENSIVE_QUESTIONS_FILE = PROJECT_ROOT / "src" / "evaluation" / "comprehensive_questions.json"

logger = setup_logging(__name__)


# ============================================================================
# PATHS (from config)
# ============================================================================

TEST_QUESTIONS_FILE = EVAL_TEST_QUESTIONS_FILE
RESULTS_DIR = EVAL_RESULTS_DIR
EVALUATION_HISTORY_FILE = PROJECT_ROOT / "memory-bank" / "evaluation-history.md"
EVALUATION_RUNS_FILE = PROJECT_ROOT / "data" / "evaluation" / "evaluation_runs.json"


# ============================================================================
# LOADER
# ============================================================================


def load_test_questions(
    filepath: Path = TEST_QUESTIONS_FILE,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load test questions from JSON file.

    Args:
        filepath: Path to test questions JSON.
        limit: Max number of questions to load.

    Returns:
        List of question dictionaries.

    Raises:
        FileNotFoundError: If test questions file doesn't exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Test questions file not found: {filepath}\n"
            "Create test questions first in data/evaluation/test_questions.json"
        )

    with open(filepath, "r") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    # Limit number of questions
    if limit and limit < len(questions):
        questions = questions[:limit]
        logger.info(f"Limited to first {limit} questions")

    return questions


# ============================================================================
# AUTO-LOGGING
# ============================================================================


def get_next_run_number() -> int:
    """
    Get the next run number from evaluation-history.md.

    Parses the markdown file to find the highest existing run number
    and returns the next sequential number.

    Returns:
        Next run number (starts at 1 if no runs exist).
    """
    if not EVALUATION_HISTORY_FILE.exists():
        return 1

    content = EVALUATION_HISTORY_FILE.read_text()
    # Match patterns like "## Run 4:" or "## Run 10:"
    matches = re.findall(r"## Run (\d+):", content)

    if not matches:
        return 1

    max_run = max(int(m) for m in matches)
    return max_run + 1


def append_to_evaluation_history(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path,
    questions: List[Dict[str, Any]],
) -> None:
    """
    Append evaluation run summary to memory-bank/evaluation-history.md.

    Auto-generates markdown entry with:
    - Run number (auto-incremented)
    - Configuration details
    - Aggregate scores
    - Category breakdown (if questions have categories)

    Args:
        results: RAGAS evaluation results with 'scores' key.
        config: Run configuration dict with collection, alpha, models, etc.
        output_path: Path where JSON results were saved.
        questions: List of test questions (for category breakdown).
    """
    run_number = get_next_run_number()

    # Calculate category breakdown
    category_breakdown = {}
    df = results.get("results")
    if df is not None:
        for i, q in enumerate(questions):
            category = q.get("category", "unknown")
            if category not in category_breakdown:
                category_breakdown[category] = {"count": 0, "relevancy_sum": 0, "faithfulness_sum": 0}
            category_breakdown[category]["count"] += 1
            # Try to get scores from DataFrame
            if i < len(df):
                relevancy = df.iloc[i].get("answer_relevancy", 0) or 0
                faithfulness = df.iloc[i].get("faithfulness", 0) or 0
                category_breakdown[category]["relevancy_sum"] += relevancy
                category_breakdown[category]["faithfulness_sum"] += faithfulness

    # Format category table
    category_table = ""
    if category_breakdown:
        category_table = "\n### Category Breakdown\n| Category | Relevancy | Faithfulness |\n|----------|-----------|--------------|\n"
        for cat, data in category_breakdown.items():
            avg_rel = data["relevancy_sum"] / data["count"] if data["count"] > 0 else 0
            avg_faith = data["faithfulness_sum"] / data["count"] if data["count"] > 0 else 0
            category_table += f"| {cat.title()} ({data['count']}) | {avg_rel:.2f} | {avg_faith:.2f} |\n"

    # Count failures (relevancy = 0)
    failures = 0
    if df is not None and "answer_relevancy" in df.columns:
        failures = len(df[df["answer_relevancy"] == 0])

    # Get relative path from project root
    try:
        relative_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_path = output_path

    # Format scores
    scores = results.get("scores", {})
    faithfulness = scores.get("faithfulness", "N/A")
    relevancy = scores.get("relevancy", "N/A")
    context_precision = scores.get("context_precision", "N/A")

    # Format score strings
    faith_str = f"{faithfulness:.3f}" if isinstance(faithfulness, (int, float)) else str(faithfulness)
    rel_str = f"{relevancy:.3f}" if isinstance(relevancy, (int, float)) else str(relevancy)
    cp_str = f"{context_precision:.3f}" if isinstance(context_precision, (int, float)) else str(context_precision)

    # Format preprocessing info
    prep_strategy = config.get('preprocessing_strategy', 'none')
    prep_model = config.get('preprocessing_model', 'default')
    prep_str = f"{prep_strategy}" + (f" ({prep_model})" if prep_model else "")

    entry = f"""
---

## Run {run_number}: {config.get('collection', 'Unknown')}

**Date:** {datetime.now().strftime('%B %d, %Y')}
**File:** `{relative_path}`

### Configuration
- **Collection:** {config.get('collection', 'auto')}
- **Search Type:** Hybrid
- **Alpha:** {config.get('alpha', 0.5)}
- **Top-K:** {config.get('top_k', 10)}
- **Reranking:** {'Yes' if config.get('reranking', False) else 'No'}
- **Preprocessing:** {prep_str}
- **Generation Model:** {config.get('generation_model', 'unknown')}
- **Evaluation Model:** {config.get('evaluation_model', 'unknown')}

### Results
| Metric | Score |
|--------|-------|
| Faithfulness | {faith_str} |
| Relevancy | {rel_str} |
| Context Precision | {cp_str} |
| Failures | {failures}/{len(questions)} ({100*failures/len(questions):.0f}%) |
{category_table}
### Key Learning
[Add notes about this run manually]

"""

    with open(EVALUATION_HISTORY_FILE, "a") as f:
        f.write(entry)

    logger.info(f"Appended to {EVALUATION_HISTORY_FILE} as Run {run_number}")


def update_tracking_json(
    results: Dict[str, Any],
    config: Dict[str, Any],
    questions: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Update data/evaluation/tracking.json with new run entry.

    Maintains a structured JSON file for programmatic analysis of
    evaluation history.

    Args:
        results: RAGAS evaluation results with 'scores' key.
        config: Run configuration dict.
        questions: List of test questions.
        output_path: Path where JSON results were saved.
    """
    # Load existing tracking data
    if EVALUATION_RUNS_FILE.exists():
        with open(EVALUATION_RUNS_FILE, "r") as f:
            tracking = json.load(f)
    else:
        tracking = {
            "_documentation": {
                "purpose": "Track evaluation runs with configurations and metrics",
                "schema_version": "1.0",
                "usage": "Updated automatically by run_stage_7_evaluation.py",
            },
            "runs": [],
            "improvements": [],
            "next_experiments": [],
        }

    # Calculate category breakdown and failures
    df = results.get("results")
    category_breakdown = {}
    failures = []

    if df is not None:
        for i, q in enumerate(questions):
            if i >= len(df):
                continue
            category = q.get("category", "unknown")
            if category not in category_breakdown:
                category_breakdown[category] = {"count": 0, "avg_relevancy": 0, "avg_faithfulness": 0}
            category_breakdown[category]["count"] += 1

            relevancy = df.iloc[i].get("answer_relevancy", 0) or 0
            faithfulness = df.iloc[i].get("faithfulness", 0) or 0

            # Track running totals for averaging
            cat = category_breakdown[category]
            n = cat["count"]
            cat["avg_relevancy"] = ((n - 1) * cat["avg_relevancy"] + relevancy) / n
            cat["avg_faithfulness"] = ((n - 1) * cat["avg_faithfulness"] + faithfulness) / n

            # Track failures
            if relevancy == 0:
                failures.append(q.get("id", f"q{i}"))

    # Round category averages
    for cat in category_breakdown.values():
        cat["avg_relevancy"] = round(cat["avg_relevancy"], 2)
        cat["avg_faithfulness"] = round(cat["avg_faithfulness"], 2)

    # Build run entry
    run_id = output_path.stem  # e.g., "eval_20251220_101507"
    scores = results.get("scores", {})

    run_entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "collection": config.get("collection", get_collection_name()),
            "search_type": "hybrid",
            "top_k": config.get("top_k", 10),
            "alpha": config.get("alpha", 0.5),
            "chunk_size": MAX_CHUNK_TOKENS,
            "overlap_sentences": OVERLAP_SENTENCES,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": config.get("generation_model", "unknown"),
            "evaluation_model": config.get("evaluation_model", "unknown"),
            "reranking": config.get("reranking", False),
            "rerank_model": "mxbai-rerank-large-v1" if config.get("reranking") else None,
            "preprocessing_strategy": config.get("preprocessing_strategy", "none"),
            "preprocessing_model": config.get("preprocessing_model"),
        },
        "metrics": {
            "faithfulness": round(scores.get("faithfulness", 0), 3),
            "relevancy": round(scores.get("relevancy", 0), 3),
            "context_precision": round(scores.get("context_precision", 0), 3) if scores.get("context_precision") else None,
            "num_questions": len(questions),
        },
        "category_breakdown": category_breakdown,
        "failures": failures,
        "notes": "",
    }

    tracking["runs"].append(run_entry)

    # Write updated tracking
    with open(EVALUATION_RUNS_FILE, "w") as f:
        json.dump(tracking, f, indent=2)

    logger.info(f"Updated {EVALUATION_RUNS_FILE} with run {run_id}")


# ============================================================================
# REPORT
# ============================================================================


def generate_report(
    results: Dict[str, Any],
    questions: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate evaluation report as JSON and summary.

    Args:
        results: RAGAS evaluation results.
        questions: Original test questions.
        output_path: Path for output JSON file.
    """
    # Create results directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "aggregate_scores": results["scores"],
        "per_question_results": [],
    }

    # Add per-question details
    df = results["results"]
    for i, q in enumerate(questions):
        question_result = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "difficulty": q["difficulty"],
        }

        # Add metric scores from DataFrame
        for col in df.columns:
            if col not in ["user_input", "retrieved_contexts", "response", "reference"]:
                question_result[col] = float(df.iloc[i][col]) if i < len(df) else None

        report["per_question_results"].append(question_result)

    # Save JSON report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nQuestions evaluated: {len(questions)}")
    print(f"\nAggregate Scores:")
    for metric, score in results["scores"].items():
        print(f"  {metric}: {score:.4f}")

    print("\nPer-Question Results:")
    for qr in report["per_question_results"]:
        print(f"\n  [{qr['id']}] {qr['question'][:50]}...")
        for key, val in qr.items():
            if key not in ["id", "question", "category", "difficulty"] and val is not None:
                print(f"    {key}: {val:.4f}")

    print("\n" + "=" * 60)


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================


def run_comprehensive_evaluation(args: argparse.Namespace) -> None:
    """Run comprehensive evaluation across all VALID combinations.

    Tests: collections x alpha values x preprocessing strategies
    - Filters out invalid combinations (e.g., graphrag + semantic collection)
    - No reranking (always disabled for speed)
    - Uses curated 10-question subset
    - Generates enhanced leaderboard report with statistical analysis
    - Outputs article-ready summary with key findings
    - No individual run logging to evaluation-history.md

    Args:
        args: Command-line arguments (uses generation_model, evaluation_model, output).
    """
    import time
    from src.ui.services.search import list_collections, extract_strategy_from_collection
    from src.rag_pipeline.retrieval.preprocessing.strategies import list_strategies
    from src.config import get_valid_preprocessing_strategies

    logger.info("Starting comprehensive evaluation mode...")
    start_time = time.time()

    # Load curated question subset
    questions = load_test_questions(filepath=COMPREHENSIVE_QUESTIONS_FILE)
    logger.info(f"Loaded {len(questions)} curated questions from comprehensive_questions.json")

    # Get all collections, alphas, all strategies
    collections = list_collections()
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    all_strategies = list_strategies()

    # Filter semantic collections to only 0.3 and 0.75 thresholds
    # Collection names: RAG_semantic_0_3_embed3large_v1 -> strategy "semantic_0.3"
    ALLOWED_SEMANTIC_THRESHOLDS = ["semantic_0.3", "semantic_0.75"]
    filtered_collections = []
    for coll in collections:
        strategy = extract_strategy_from_collection(coll)
        if strategy.startswith("semantic_"):
            if strategy in ALLOWED_SEMANTIC_THRESHOLDS:
                filtered_collections.append(coll)
            else:
                logger.info(f"Skipping collection: {coll} (strategy '{strategy}' not in allowed thresholds)")
        else:
            # Non-semantic collections pass through
            filtered_collections.append(coll)
    collections = filtered_collections

    if not collections:
        logger.error("No RAG collections found in Weaviate. Run stage 6 first.")
        return

    # Build valid combinations list (filter out invalid ones like graphrag + semantic)
    valid_combinations = []
    for collection in collections:
        collection_strategy = extract_strategy_from_collection(collection)
        valid_preprocessing = get_valid_preprocessing_strategies(collection_strategy)
        for alpha in alphas:
            for strategy in all_strategies:
                if strategy in valid_preprocessing:
                    valid_combinations.append((collection, alpha, strategy))

    # Log what we're testing
    skipped = len(collections) * len(alphas) * len(all_strategies) - len(valid_combinations)
    logger.info(f"Testing {len(valid_combinations)} valid combinations ({skipped} invalid skipped):")
    logger.info(f"  Collections ({len(collections)}): {collections}")
    logger.info(f"  Alphas ({len(alphas)}): {alphas}")
    logger.info(f"  Strategies ({len(all_strategies)}): {all_strategies}")
    logger.info(f"  Note: graphrag only valid with section/contextual collections")

    # Run all valid combinations
    all_results = []
    count = 0

    for collection, alpha, strategy in valid_combinations:
        count += 1
        logger.info(
            f"\n[{count}/{len(valid_combinations)}] "
            f"{collection} | alpha={alpha} | strategy={strategy}"
        )

        try:
            results = run_evaluation(
                test_questions=questions,
                metrics=["faithfulness", "relevancy", "context_precision"],
                top_k=getattr(args, 'top_k', DEFAULT_TOP_K),
                generation_model=args.generation_model,
                evaluation_model=args.evaluation_model,
                collection_name=collection,
                use_reranking=False,  # Always disabled for speed
                alpha=alpha,
                preprocessing_strategy=strategy,
                preprocessing_model=None,
            )

            # Store configuration + scores
            all_results.append({
                "collection": collection,
                "alpha": alpha,
                "strategy": strategy,
                "scores": results["scores"],
                "num_questions": len(questions),
            })

            scores = results["scores"]
            logger.info(
                f"  -> faith={scores.get('faithfulness', 0):.3f} "
                f"relev={scores.get('relevancy', 0):.3f} "
                f"ctx_prec={scores.get('context_precision', 0):.3f}"
            )

        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            all_results.append({
                "collection": collection,
                "alpha": alpha,
                "strategy": strategy,
                "scores": {"faithfulness": 0, "relevancy": 0, "context_precision": 0},
                "num_questions": len(questions),
                "error": str(e),
            })

    # Calculate total duration
    duration_seconds = time.time() - start_time

    # Generate comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"comprehensive_{timestamp}.json"

    # Build grid parameters for report
    grid_params = {
        "collections": collections,
        "alphas": alphas,
        "strategies": all_strategies,
        "valid_combinations_count": len(valid_combinations),
        "skipped_count": skipped,
    }

    generate_comprehensive_report(
        all_results, questions, output_path, duration_seconds, grid_params
    )

    logger.info("Comprehensive evaluation complete!")


def compute_statistical_breakdown(
    results: List[Dict[str, Any]],
    group_key: str,
) -> Dict[str, Dict[str, float]]:
    """Compute statistical analysis (mean, std, min, max) for each group.

    Args:
        results: List of result dicts (only successful runs).
        group_key: Key to group by ("strategy", "alpha", "collection").

    Returns:
        Dict mapping group values to stats (mean, std, min, max, n).
    """
    import statistics
    from collections import defaultdict

    groups: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        if "error" not in r:
            groups[str(r[group_key])].append(r["composite_score"])

    analysis = {}
    for group, scores in groups.items():
        analysis[group] = {
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "n": len(scores),
        }
    return analysis


def find_best_configurations(
    sorted_results: List[Dict[str, Any]],
    successful_runs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Find best configurations overall and per metric.

    Args:
        sorted_results: Results sorted by composite score.
        successful_runs: Only successful runs.

    Returns:
        Dict with "overall" and per-metric best configurations.
    """
    best = {
        "overall": {
            "collection": sorted_results[0]["collection"],
            "alpha": sorted_results[0]["alpha"],
            "strategy": sorted_results[0]["strategy"],
            "composite_score": round(sorted_results[0]["composite_score"], 4),
        } if sorted_results else None,
    }

    # Find best per metric
    for metric in ["faithfulness", "relevancy", "context_precision"]:
        sorted_by_metric = sorted(
            successful_runs,
            key=lambda x: x["scores"].get(metric) or 0,
            reverse=True,
        )
        if sorted_by_metric:
            top = sorted_by_metric[0]
            best[f"by_{metric}"] = {
                "collection": top["collection"],
                "alpha": top["alpha"],
                "strategy": top["strategy"],
                "score": round(top["scores"].get(metric) or 0, 4),
            }

    return best


def generate_comprehensive_report(
    all_results: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    output_path: Path,
    duration_seconds: float = 0.0,
    grid_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate enhanced leaderboard and statistical analysis for articles.

    Produces:
    - JSON report with experiment metadata, statistical analysis, best configs
    - Console leaderboard with rankings
    - Article-ready summary with key findings

    Args:
        all_results: List of result dicts with collection, alpha, strategy, scores.
        questions: Original test questions.
        output_path: Path for output JSON file.
        duration_seconds: Total experiment duration in seconds.
        grid_params: Grid search parameters (collections, alphas, strategies).
    """
    # Create results directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate composite score for ranking (average of all metrics)
    for result in all_results:
        scores = result["scores"]
        faithfulness = scores.get("faithfulness") or 0
        relevancy = scores.get("relevancy") or 0
        context_precision = scores.get("context_precision") or 0
        result["composite_score"] = (faithfulness + relevancy + context_precision) / 3

    # Sort by composite score (descending)
    sorted_results = sorted(
        all_results,
        key=lambda x: x["composite_score"],
        reverse=True
    )

    # Separate successful and failed runs
    successful_runs = [r for r in all_results if "error" not in r]
    failed_runs = [r for r in all_results if "error" in r]

    # Compute statistical analysis
    strategy_analysis = compute_statistical_breakdown(successful_runs, "strategy")
    alpha_analysis = compute_statistical_breakdown(successful_runs, "alpha")
    collection_analysis = compute_statistical_breakdown(successful_runs, "collection")

    # Find best configurations
    best_configs = find_best_configurations(sorted_results, successful_runs)

    # Format duration
    duration_minutes = duration_seconds / 60
    duration_str = f"{int(duration_minutes)} minutes {int(duration_seconds % 60)} seconds"

    # Build enhanced report
    report = {
        "experiment_metadata": {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "duration_formatted": duration_str,
            "total_combinations": len(all_results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
        },
        "grid_parameters": {
            "collections": grid_params.get("collections", []) if grid_params else [],
            "alphas": grid_params.get("alphas", []) if grid_params else [],
            "strategies": grid_params.get("strategies", []) if grid_params else [],
            "num_questions": len(questions),
            "reranking_enabled": False,
        },
        "leaderboard": sorted_results,
        "statistical_analysis": {
            "by_strategy": strategy_analysis,
            "by_alpha": alpha_analysis,
            "by_collection": collection_analysis,
        },
        "best_configurations": best_configs,
        "failed_runs": [
            {
                "collection": r["collection"],
                "alpha": r["alpha"],
                "strategy": r["strategy"],
                "error": r.get("error", "Unknown"),
            }
            for r in failed_runs
        ],
    }

    # Save JSON report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Comprehensive report saved to: {output_path}")

    # =========================================================================
    # CONSOLE OUTPUT: Leaderboard
    # =========================================================================
    print("\n" + "=" * 100)
    print("COMPREHENSIVE EVALUATION LEADERBOARD")
    print("=" * 100)
    print(f"\nTested {len(all_results)} combinations on {len(questions)} questions")
    print(f"Duration: {duration_str}")
    print(f"Successful: {len(successful_runs)} | Failed: {len(failed_runs)}")
    print("\n" + "-" * 100)
    print(f"{'Rank':<5} {'Collection':<35} {'Alpha':<7} {'Strategy':<15} {'Faith':<8} {'Relev':<8} {'CtxPrec':<8} {'Avg':<8}")
    print("-" * 100)

    for i, result in enumerate(sorted_results, 1):
        scores = result["scores"]
        faith = scores.get("faithfulness") or 0
        relev = scores.get("relevancy") or 0
        ctx_prec = scores.get("context_precision") or 0
        avg = result["composite_score"]
        collection_short = result["collection"][:35]
        error_marker = " *" if "error" in result else ""

        print(
            f"{i:<5} {collection_short:<35} {result['alpha']:<7} "
            f"{result['strategy']:<15} {faith:<8.3f} {relev:<8.3f} "
            f"{ctx_prec:<8.3f} {avg:<8.3f}{error_marker}"
        )

    # =========================================================================
    # CONSOLE OUTPUT: Statistical Breakdowns
    # =========================================================================
    def _print_breakdown(title: str, analysis: Dict[str, Dict[str, float]], format_label=str):
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        for group in sorted(analysis.keys(), key=lambda x: analysis[x]["mean"], reverse=True):
            stats = analysis[group]
            label = format_label(group)
            print(f"\n{label} (n={stats['n']}):")
            print(f"  Mean:  {stats['mean']:.3f}  +/-  {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")

    _print_breakdown("BREAKDOWN BY PREPROCESSING STRATEGY", strategy_analysis, str.upper)
    _print_breakdown("BREAKDOWN BY ALPHA (0.0=keyword, 1.0=vector)", alpha_analysis, lambda a: f"ALPHA={a}")
    _print_breakdown("BREAKDOWN BY COLLECTION", collection_analysis)

    # =========================================================================
    # CONSOLE OUTPUT: Article Summary
    # =========================================================================
    print("\n" + "=" * 100)
    print("ARTICLE SUMMARY")
    print("=" * 100)

    print(f"\nEXPERIMENT METADATA")
    print(f"  Duration: {duration_str}")
    print(f"  Combinations tested: {len(all_results)} ({len(successful_runs)} successful, {len(failed_runs)} failed)")
    print(f"  Questions per run: {len(questions)}")

    print(f"\nBEST CONFIGURATIONS")
    if best_configs.get("overall"):
        b = best_configs["overall"]
        print(f"  Overall:        {b['collection'][:30]} + alpha={b['alpha']} + {b['strategy']} (avg: {b['composite_score']:.3f})")
    for metric in ["faithfulness", "relevancy", "context_precision"]:
        key = f"by_{metric}"
        if best_configs.get(key):
            b = best_configs[key]
            print(f"  {metric.title():<15} {b['collection'][:30]} + alpha={b['alpha']} + {b['strategy']} ({b['score']:.3f})")

    # Calculate improvement percentages vs baseline (none strategy)
    if "none" in strategy_analysis:
        baseline = strategy_analysis["none"]["mean"]
        print(f"\nSTRATEGY IMPROVEMENT VS BASELINE (none={baseline:.3f})")
        for strategy in sorted(strategy_analysis.keys()):
            stats = strategy_analysis[strategy]
            if strategy == "none":
                print(f"  {strategy.upper():<15} {stats['mean']:.3f} +/- {stats['std']:.3f}  (baseline)")
            else:
                improvement = ((stats["mean"] - baseline) / baseline) * 100 if baseline > 0 else 0
                print(f"  {strategy.upper():<15} {stats['mean']:.3f} +/- {stats['std']:.3f}  [{improvement:+.1f}% vs baseline]")

    # Show failed runs if any
    if failed_runs:
        print("\n" + "=" * 100)
        print("FAILED RUNS")
        print("=" * 100)
        for result in failed_runs:
            print(f"  {result['collection']} | alpha={result['alpha']} | {result['strategy']}")
            print(f"    Error: {result.get('error', 'Unknown')}")

    print("\n" + "=" * 100)


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run RAGAS evaluation on test questions."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on test questions"
    )
    parser.add_argument(
        "--questions",
        "-n",
        type=int,
        default=None,
        help="Limit to first N questions",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["faithfulness", "relevancy", "context_precision"],
        help="Metrics to compute (default: faithfulness relevancy context_precision)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=EVAL_GENERATION_MODEL,
        help=f"OpenRouter model for answer generation (default: {EVAL_GENERATION_MODEL})",
    )
    parser.add_argument(
        "--evaluation-model",
        type=str,
        default=EVAL_EVALUATION_MODEL,
        help=f"OpenRouter model for RAGAS evaluation (default: {EVAL_EVALUATION_MODEL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: results/eval_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.5,
        help="Hybrid search alpha: 0.0=keyword, 0.5=balanced, 1.0=vector (default: 0.5)",
    )
    parser.add_argument(
        "--reranking",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking",
    )
    parser.add_argument(
        "--no-reranking",
        dest="reranking",
        action="store_false",
        help="Disable cross-encoder reranking (default)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Weaviate collection to evaluate (default: auto from config)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        default=False,
        help="Skip auto-logging to evaluation-history.md and tracking.json",
    )
    parser.add_argument(
        "--preprocessing",
        "-p",
        type=str,
        choices=["none", "hyde", "decomposition", "graphrag"],
        default="none",
        help="Query preprocessing strategy (default: none for clean baseline)",
    )
    parser.add_argument(
        "--preprocessing-model",
        type=str,
        default=None,
        help="Model for preprocessing (default: from config)",
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        choices=["prompt", "skip", "all"],
        default="prompt",
        help="Overwrite behavior for custom -o path: prompt (default), skip, all",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        default=False,
        help="Run comprehensive evaluation across all collections, alphas (0.0-1.0), and strategies",
    )

    args = parser.parse_args()

    # Comprehensive mode: different execution path
    if args.comprehensive:
        run_comprehensive_evaluation(args)
        return

    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    # Check overwrite for custom output path before doing expensive work
    if args.output:
        output_path = Path(args.output)
        if output_path.exists():
            if not overwrite_context.should_overwrite(output_path, logger):
                logger.info("Skipping evaluation (output file exists)")
                return

    # Load test questions
    logger.info("Loading test questions...")
    questions = load_test_questions(
        limit=args.questions,
    )

    if not questions:
        logger.error("No test questions found")
        return

    logger.info(f"Loaded {len(questions)} test questions")

    # Determine collection name
    collection_name = args.collection or get_collection_name()

    # Run evaluation
    logger.info("Starting RAGAS evaluation...")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Reranking: {args.reranking}")
    logger.info(f"Preprocessing: {args.preprocessing}")
    logger.info(f"Generation model: {args.generation_model}")
    logger.info(f"Evaluation model: {args.evaluation_model}")

    try:
        results = run_evaluation(
            test_questions=questions,
            metrics=args.metrics,
            top_k=args.top_k,
            generation_model=args.generation_model,
            evaluation_model=args.evaluation_model,
            collection_name=collection_name,
            use_reranking=args.reranking,
            alpha=args.alpha,
            preprocessing_strategy=args.preprocessing,
            preprocessing_model=args.preprocessing_model,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_{timestamp}.json"

    generate_report(results, questions, output_path)

    # Auto-log results to evaluation history and tracking JSON
    if not args.no_log:
        config = {
            "collection": collection_name,
            "alpha": args.alpha,
            "top_k": args.top_k,
            "reranking": args.reranking,
            "preprocessing_strategy": args.preprocessing,
            "preprocessing_model": args.preprocessing_model,
            "generation_model": args.generation_model,
            "evaluation_model": args.evaluation_model,
        }
        append_to_evaluation_history(results, config, output_path, questions)
        update_tracking_json(results, config, questions, output_path)
    else:
        logger.info("Skipping auto-logging (--no-log specified)")

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
