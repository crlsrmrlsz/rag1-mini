"""Stage 7C: RAGAS Evaluation for RAG1-Mini.

Runs RAGAS evaluation on test questions to measure RAG pipeline quality.

Usage:
    python -m src.run_stage_7_evaluation
    python -m src.run_stage_7_evaluation --questions 5  # Run on first 5 questions
    python -m src.run_stage_7_evaluation --metrics faithfulness relevancy

Prerequisites:
    - Weaviate must be running (docker compose up -d)
    - Stage 6 must have been run to populate the collection
    - OpenRouter API key must be set in .env
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import PROJECT_ROOT, DEFAULT_TOP_K
from src.evaluation import run_evaluation
from src.utils.file_utils import setup_logging

logger = setup_logging(__name__)


# ============================================================================
# PATHS
# ============================================================================

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
TEST_QUESTIONS_FILE = EVALUATION_DIR / "test_questions.json"
RESULTS_DIR = EVALUATION_DIR / "results"


# ============================================================================
# LOADER
# ============================================================================


def load_test_questions(
    filepath: Path = TEST_QUESTIONS_FILE,
    limit: Optional[int] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load test questions from JSON file.

    Args:
        filepath: Path to test questions JSON.
        limit: Max number of questions to load.
        category: Filter by category (neuroscience, philosophy, synthesis, open_ended).

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

    # Filter by category if specified
    if category:
        questions = [q for q in questions if q.get("category") == category]
        logger.info(f"Filtered to {len(questions)} questions in category: {category}")

    # Limit number of questions
    if limit and limit < len(questions):
        questions = questions[:limit]
        logger.info(f"Limited to first {limit} questions")

    return questions


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
        "--category",
        "-c",
        type=str,
        choices=["neuroscience", "philosophy", "synthesis", "open_ended"],
        default=None,
        help="Filter by question category",
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
        default="openai/gpt-4o-mini",
        help="OpenRouter model for answer generation",
    )
    parser.add_argument(
        "--evaluation-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model for RAGAS evaluation",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: results/eval_TIMESTAMP.json)",
    )

    args = parser.parse_args()

    # Load test questions
    logger.info("Loading test questions...")
    questions = load_test_questions(
        limit=args.questions,
        category=args.category,
    )

    if not questions:
        logger.error("No test questions found")
        return

    logger.info(f"Loaded {len(questions)} test questions")

    # Run evaluation
    logger.info("Starting RAGAS evaluation...")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Generation model: {args.generation_model}")
    logger.info(f"Evaluation model: {args.evaluation_model}")

    try:
        results = run_evaluation(
            test_questions=questions,
            metrics=args.metrics,
            top_k=args.top_k,
            generation_model=args.generation_model,
            evaluation_model=args.evaluation_model,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_{timestamp}.json"

    generate_report(results, questions, output_path)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
