"""RAGAS evaluation for RAGLab.

Provides:
- OpenRouter chat integration for answer generation
- RAGAS evaluation with LangChain wrapper
- Retrieval and generation functions for the evaluation pipeline
- Strategy-aware retrieval (decomposition RRF, graphrag hybrid)
"""

import time
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.rag_pipeline.retrieval.preprocessing.query_preprocessing import PreprocessedQuery

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    FactualCorrectness,
    AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL_ID,
    get_collection_name,
    DEFAULT_TOP_K,
    EVAL_TRACES_DIR,
)
from src.rag_pipeline.indexing import get_client, query_hybrid
from src.rag_pipeline.retrieval.reranking_utils import apply_reranking_if_enabled
from src.shared.files import setup_logging
from src.shared.openrouter_client import call_simple_prompt
from src.evaluation.schemas import QuestionTrace, EvaluationTrace

logger = setup_logging(__name__)


# Default model for answer generation (can be overridden)
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"

# Mapping from our metric names to RAGAS DataFrame column names
RAGAS_METRIC_COLUMN_MAP = {
    "faithfulness": "faithfulness",
    "relevancy": "answer_relevancy",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "answer_correctness": "answer_correctness",
}


def _is_valid_score(value: Any) -> bool:
    """Check if a value is a valid numeric score (not None, not NaN).

    Args:
        value: The value to check.

    Returns:
        True if the value is a valid numeric score.
    """
    if value is None:
        return False
    try:
        float_val = float(value)
        return not math.isnan(float_val)
    except (TypeError, ValueError):
        return False




# ============================================================================
# RAGAS LLM WRAPPER
# ============================================================================


def create_evaluator_llm(model: str = "openai/gpt-4o-mini") -> LangchainLLMWrapper:
    """
    Create LLM wrapper for RAGAS evaluation via OpenRouter.

    Args:
        model: OpenRouter model ID for evaluation.

    Returns:
        LangchainLLMWrapper configured for OpenRouter.
    """
    llm = ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
    )
    return LangchainLLMWrapper(llm)


def create_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """
    Create embeddings wrapper for RAGAS evaluation via OpenRouter.

    Returns:
        LangchainEmbeddingsWrapper configured for OpenRouter.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_ID,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    return LangchainEmbeddingsWrapper(embeddings)


class RAGASEvaluationError(Exception):
    """Raised when RAGAS evaluation fails after all retries."""

    pass


def evaluate_with_retry(
    dataset: EvaluationDataset,
    metrics: List,
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> Any:
    """Run RAGAS evaluation with exponential backoff retry.

    Wraps ragas.evaluate() to handle transient API failures.
    Uses the same retry pattern as openrouter_client.py.

    Args:
        dataset: RAGAS EvaluationDataset.
        metrics: List of RAGAS metric objects.
        llm: Wrapped LLM for evaluation.
        embeddings: Wrapped embeddings for evaluation.
        max_retries: Maximum retry attempts.
        backoff_base: Exponential backoff base.

    Returns:
        RAGAS evaluation results.

    Raises:
        RAGASEvaluationError: After all retries exhausted.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )
            return results

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if it's a retryable error (rate limit, server error, network)
            is_retryable = (
                "rate" in error_str
                or "limit" in error_str
                or "429" in error_str
                or "500" in error_str
                or "502" in error_str
                or "503" in error_str
                or "timeout" in error_str
                or "connection" in error_str
            )

            if is_retryable and attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(
                    f"RAGAS evaluation failed ({type(e).__name__}), "
                    f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            # Non-retryable error or max retries exceeded
            break

    raise RAGASEvaluationError(
        f"RAGAS evaluation failed after {max_retries} retries: {last_error}"
    ) from last_error


def compute_difficulty_breakdown(
    results_df: Any,
    test_questions: List[Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-difficulty metric averages.

    Groups questions by difficulty field ("single_concept" or "cross_domain")
    and computes average scores for each metric within each group.

    Args:
        results_df: RAGAS results DataFrame with metric columns.
        test_questions: Original test questions with difficulty field.
        metrics: List of metric names to include in breakdown.

    Returns:
        Dict mapping difficulty -> {metric: avg_score}.
        Example: {
            "single_concept": {"faithfulness": 0.95, "relevancy": 0.88},
            "cross_domain": {"faithfulness": 0.82, "relevancy": 0.75}
        }
    """
    breakdown: Dict[str, Dict[str, List[float]]] = {}

    for i, q in enumerate(test_questions):
        if i >= len(results_df):
            continue

        difficulty = q.get("difficulty", "unknown")
        if difficulty not in breakdown:
            breakdown[difficulty] = {m: [] for m in metrics}

        for metric in metrics:
            col_name = RAGAS_METRIC_COLUMN_MAP.get(metric, metric)
            if col_name in results_df.columns:
                value = results_df.iloc[i].get(col_name)
                if _is_valid_score(value):
                    breakdown[difficulty][metric].append(float(value))

    # Compute averages
    result = {}
    for difficulty, metric_scores in breakdown.items():
        result[difficulty] = {}
        for metric, scores in metric_scores.items():
            if scores:
                result[difficulty][metric] = round(sum(scores) / len(scores), 4)
            else:
                result[difficulty][metric] = 0.0

    return result


# ============================================================================
# RETRIEVAL AND GENERATION
# ============================================================================


def retrieve_contexts(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
    preprocessed: Optional[PreprocessedQuery] = None,
) -> List[str]:
    """
    Retrieve relevant contexts from Weaviate for a question.

    This function implements strategy-aware retrieval:
    1. For decomposition: Executes each sub-query and merges with RRF
    2. For graphrag: Combines vector search with Neo4j graph traversal
    3. For other strategies: Standard hybrid search with optional reranking

    Args:
        question: The user's question (or preprocessed search_query).
        top_k: Number of chunks to return after reranking.
        collection_name: Override collection name.
        use_reranking: If True, apply cross-encoder reranking.
                       Retrieves 50 candidates and reranks to top_k.
                       If False, directly returns top_k from hybrid search.
        alpha: Hybrid search balance (0.0=keyword, 0.5=balanced, 1.0=vector).
        preprocessed: PreprocessedQuery from strategy, enables strategy-aware
                     retrieval for decomposition (RRF) and graphrag (Neo4j).

    Returns:
        List of context strings from retrieved chunks.

    Technical Notes:
        - Hybrid search combines vector similarity with BM25 keyword matching
        - alpha=0.0 is pure BM25, alpha=1.0 is pure vector, 0.5 is balanced
        - Decomposition uses RRF to merge results from 3-4 sub-queries
        - GraphRAG boosts chunks found via Neo4j entity traversal
        - Cross-encoder reranking improves precision by 20-35%
    """
    collection_name = collection_name or get_collection_name()

    # Determine initial retrieval size
    initial_k = 50 if use_reranking else top_k

    # =========================================================================
    # DECOMPOSITION: Multi-query with RRF merge
    # =========================================================================
    if preprocessed and preprocessed.strategy_used == "decomposition":
        generated_queries = preprocessed.generated_queries or []
        if len(generated_queries) > 1:
            logger.info(f"  [decomposition] Executing {len(generated_queries)} queries with RRF merge")

            # Import RRF infrastructure from UI search service
            from src.rag_pipeline.retrieval.rrf import reciprocal_rank_fusion

            client = get_client()
            try:
                result_lists = []
                query_types = []

                # Execute each sub-query
                per_query_k = max(top_k * 2, 20)
                for q in generated_queries:
                    query_text = q.get("query", "")
                    query_type = q.get("type", "unknown")

                    if not query_text:
                        continue

                    results = query_hybrid(
                        client=client,
                        query_text=query_text,
                        top_k=per_query_k,
                        alpha=alpha,
                        collection_name=collection_name,
                    )

                    result_lists.append(results)
                    query_types.append(query_type)

                # Merge with RRF
                rrf_result = reciprocal_rank_fusion(
                    result_lists=result_lists,
                    query_types=query_types,
                    top_k=initial_k,
                )
                results = rrf_result.results

                # Apply reranking if enabled (using helper for consistency)
                results = apply_reranking_if_enabled(
                    results, preprocessed.original_query, top_k, use_reranking
                )

                return [r.text for r in results]

            finally:
                client.close()

    # =========================================================================
    # GRAPHRAG: Hybrid graph + vector retrieval
    # =========================================================================
    if preprocessed and preprocessed.strategy_used == "graphrag":
        logger.info("  [graphrag] Executing hybrid graph + vector retrieval")

        client = get_client()
        try:
            # First, get vector results
            vector_results = query_hybrid(
                client=client,
                query_text=question,
                top_k=initial_k,
                alpha=alpha,
                collection_name=collection_name,
            )

            # Convert SearchResult to dicts for hybrid_graph_retrieval
            vector_dicts = [
                {
                    "chunk_id": r.chunk_id,
                    "book_id": r.book_id,
                    "section": r.section,
                    "context": r.context,
                    "text": r.text,
                    "token_count": r.token_count,
                    "similarity": r.score,
                }
                for r in vector_results
            ]

            # Merge with Neo4j graph traversal
            try:
                from src.graph.neo4j_client import get_driver
                from src.graph.query import hybrid_graph_retrieval

                driver = get_driver()
                try:
                    merged_results, graph_meta = hybrid_graph_retrieval(
                        query=preprocessed.original_query,
                        driver=driver,
                        vector_results=vector_dicts,
                        top_k=initial_k,
                    )

                    boosted_count = graph_meta.get("boosted_count", 0)
                    logger.info(f"  [graphrag] {boosted_count} graph-boosted results")

                    # Apply reranking if enabled (on merged results)
                    if use_reranking and merged_results:
                        # Convert dicts back to SearchResult for reranker
                        from src.rag_pipeline.indexing.weaviate_query import SearchResult
                        rerank_input = [
                            SearchResult(
                                chunk_id=r.get("chunk_id", ""),
                                book_id=r.get("book_id", ""),
                                section=r.get("section", ""),
                                context=r.get("context", ""),
                                text=r.get("text", ""),
                                token_count=r.get("token_count", 0),
                                score=r.get("similarity", 0.0),
                            )
                            for r in merged_results
                        ]
                        reranked = apply_reranking_if_enabled(
                            rerank_input, preprocessed.original_query, top_k, use_reranking
                        )
                        return [r.text for r in reranked]

                    return [r.get("text", "") for r in merged_results[:top_k]]

                finally:
                    driver.close()

            except Exception as e:
                # Fallback to vector-only if Neo4j fails
                logger.warning(f"  [graphrag] Neo4j retrieval failed: {e}, using vector-only")
                vector_results = apply_reranking_if_enabled(
                    vector_results, preprocessed.original_query, top_k, use_reranking
                )
                return [r.text for r in vector_results[:top_k]]

        finally:
            client.close()

    # =========================================================================
    # HYDE: K=5 hypotheticals with embedding averaging (paper recommendation)
    # =========================================================================
    if preprocessed and preprocessed.strategy_used == "hyde":
        generated_queries = preprocessed.generated_queries or []
        hyde_passages = [q.get("query", "") for q in generated_queries if q.get("type") == "hyde" and q.get("query")]

        if len(hyde_passages) > 1:
            logger.info(f"  [hyde] Averaging {len(hyde_passages)} hypothetical embeddings")

            from src.rag_pipeline.embedding.embedder import embed_texts

            # Embed all K hypothetical passages
            embeddings = embed_texts(hyde_passages)  # List[List[float]]

            # Average embeddings (element-wise mean)
            avg_embedding = [sum(col) / len(col) for col in zip(*embeddings)]

            client = get_client()
            try:
                # Use averaged embedding for search, original query for BM25
                results = query_hybrid(
                    client=client,
                    query_text=preprocessed.original_query,  # Original for BM25
                    top_k=initial_k,
                    alpha=alpha,
                    collection_name=collection_name,
                    precomputed_embedding=avg_embedding,  # Averaged for vector
                )

                results = apply_reranking_if_enabled(
                    results, preprocessed.original_query, top_k, use_reranking
                )

                return [r.text for r in results]
            finally:
                client.close()

    # =========================================================================
    # DEFAULT: Standard hybrid search (for none and fallback)
    # =========================================================================
    client = get_client()

    try:
        results = query_hybrid(
            client=client,
            query_text=question,
            top_k=initial_k,
            alpha=alpha,
            collection_name=collection_name,
        )

        # Apply cross-encoder reranking if enabled (using helper for consistency)
        results = apply_reranking_if_enabled(results, question, top_k, use_reranking)

        return [r.text for r in results]
    finally:
        client.close()


def generate_answer(
    question: str,
    contexts: List[str],
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Generate an answer using retrieved contexts.

    Args:
        question: The user's question.
        contexts: List of context strings from retrieval.
        model: OpenRouter model ID for generation.

    Returns:
        Generated answer string.
    """
    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""Based on the following context, answer the question.
Only use information from the context. If the context doesn't contain
enough information to fully answer the question, say so explicitly.

Context:
{context_text}

Question: {question}

Answer:"""

    return call_simple_prompt(prompt, model=model)


# ============================================================================
# RAGAS EVALUATION
# ============================================================================


def run_evaluation(
    test_questions: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    generation_model: str = DEFAULT_CHAT_MODEL,
    evaluation_model: str = "openai/gpt-4o-mini",
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
    preprocessing_strategy: str = "none",
    preprocessing_model: Optional[str] = None,
    save_trace: bool = True,
    trace_path: Optional[Path] = None,
    ragas_max_retries: int = 3,
    ragas_backoff_base: float = 2.0,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on test questions with traceability and resilience.

    This function:
    1. Optionally preprocesses each question (hyde, decomposition, etc.)
    2. Retrieves contexts for each question (with optional cross-encoder reranking)
    3. Generates answers using the RAG pipeline
    4. Evaluates using RAGAS metrics with retry logic
    5. Builds and saves trace file for recalculation

    Args:
        test_questions: List of dicts with 'question' and optionally 'reference' keys.
        metrics: Which metrics to compute. Options:
            - "faithfulness": Is the answer grounded in context?
            - "relevancy": Does the answer address the question?
            - "context_precision": Are retrieved chunks relevant?
            - "context_recall": Did retrieval capture needed info? (requires reference)
            - "answer_correctness": Is the answer factually correct? (requires reference)
        top_k: Number of chunks to retrieve per question.
        generation_model: Model for answer generation.
        evaluation_model: Model for RAGAS evaluation.
        collection_name: Override Weaviate collection.
        use_reranking: If True, apply cross-encoder reranking to improve retrieval.
                       Default: True (enabled for best accuracy).
        alpha: Hybrid search balance (0.0=keyword, 0.5=balanced, 1.0=vector).
        preprocessing_strategy: Query preprocessing strategy ("none", "hyde", "decomposition").
                               Default: "none" for clean baseline evaluation.
        preprocessing_model: Model for preprocessing LLM calls (default from config).
        save_trace: If True, save trace file with all interactions for recalculation.
        trace_path: Custom path for trace file. If None, auto-generates path.
        ragas_max_retries: Maximum retries for RAGAS evaluation API calls.
        ragas_backoff_base: Exponential backoff base for RAGAS retries.

    Returns:
        Dict with:
            - "scores": Dict of metric_name -> average score
            - "results": Per-question results DataFrame
            - "samples": List of evaluation samples
            - "difficulty_breakdown": Per-difficulty group metric averages
            - "trace_path": Path where trace was saved (if save_trace=True)

    Raises:
        RAGASEvaluationError: If RAGAS evaluation fails after all retries.
    """
    from src.config import EVAL_DEFAULT_METRICS

    # Default metrics from centralized config
    if metrics is None:
        metrics = EVAL_DEFAULT_METRICS.copy()

    # Generate run_id for trace
    run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    resolved_collection = collection_name or get_collection_name()

    logger.info(f"Starting evaluation with {len(test_questions)} questions")
    logger.info(f"Metrics: {metrics}")

    # Build config dict for trace
    config = {
        "collection": resolved_collection,
        "alpha": alpha,
        "top_k": top_k,
        "use_reranking": use_reranking,
        "preprocessing_strategy": preprocessing_strategy,
        "preprocessing_model": preprocessing_model,
        "generation_model": generation_model,
        "evaluation_model": evaluation_model,
    }

    # Build evaluation samples and question traces
    samples = []
    question_traces = []

    for i, q in enumerate(test_questions):
        question = q["question"]
        reference = q.get("reference")
        question_id = q.get("id", f"q_{i}")
        difficulty = q.get("difficulty", "unknown")
        category = q.get("category", "unknown")

        logger.info(f"Processing question {i + 1}/{len(test_questions)}: {question[:50]}...")

        # Apply preprocessing if enabled
        search_query = question
        preprocessed = None
        generated_queries = None

        if preprocessing_strategy != "none":
            from src.rag_pipeline.retrieval.preprocessing import preprocess_query
            try:
                preprocessed = preprocess_query(
                    query=question,
                    strategy=preprocessing_strategy,
                    model=preprocessing_model,
                )
                search_query = preprocessed.search_query
                generated_queries = preprocessed.generated_queries
                logger.info(f"  Preprocessed ({preprocessing_strategy}): {search_query[:50]}...")
            except Exception as e:
                logger.warning(f"  Preprocessing failed: {e}. Using original query.")
                search_query = question
                preprocessed = None

        # Retrieve contexts with strategy-aware routing
        contexts = retrieve_contexts(
            question=search_query,
            top_k=top_k,
            collection_name=resolved_collection,
            use_reranking=use_reranking,
            alpha=alpha,
            preprocessed=preprocessed,
        )
        logger.info(f"  Retrieved {len(contexts)} contexts")

        # Generate answer
        answer = generate_answer(
            question=question,
            contexts=contexts,
            model=generation_model,
        )
        logger.info(f"  Generated answer: {answer[:100]}...")

        # Build RAGAS sample
        sample = {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if reference:
            sample["reference"] = reference
        samples.append(sample)

        # Build question trace
        question_trace = QuestionTrace(
            question_id=question_id,
            question=question,
            difficulty=difficulty,
            category=category,
            reference=reference,
            preprocessing_strategy=preprocessing_strategy,
            search_query=search_query,
            generated_queries=generated_queries,
            retrieved_contexts=contexts,
            retrieval_metadata={
                "top_k": top_k,
                "alpha": alpha,
                "collection": resolved_collection,
                "reranking": use_reranking,
            },
            generated_answer=answer,
            generation_model=generation_model,
        )
        question_traces.append(question_trace)

    # Create RAGAS dataset
    dataset = EvaluationDataset.from_list(samples)

    # Map metric names to objects
    metric_map = {
        "faithfulness": Faithfulness(),
        "relevancy": ResponseRelevancy(),
        "context_precision": LLMContextPrecisionWithoutReference(),
        "context_recall": LLMContextRecall(),
        "answer_correctness": AnswerCorrectness(),
    }

    # Validate metrics
    selected_metrics = []
    selected_metric_names = []
    for m in metrics:
        if m not in metric_map:
            logger.warning(f"Unknown metric: {m}, skipping")
            continue

        # Check if metric requires reference
        if m in ["context_recall", "answer_correctness"]:
            has_references = all(q.get("reference") for q in test_questions)
            if not has_references:
                logger.warning(f"Metric {m} requires reference answers, skipping")
                continue

        selected_metrics.append(metric_map[m])
        selected_metric_names.append(m)

    if not selected_metrics:
        raise ValueError("No valid metrics selected")

    logger.info(f"Running RAGAS evaluation with {len(selected_metrics)} metrics...")

    # Create evaluator LLM and embeddings
    evaluator_llm = create_evaluator_llm(model=evaluation_model)
    evaluator_embeddings = create_evaluator_embeddings()

    # Run evaluation with retry logic
    results = evaluate_with_retry(
        dataset=dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        max_retries=ragas_max_retries,
        backoff_base=ragas_backoff_base,
    )

    logger.info("Evaluation complete")

    # Convert to DataFrame for analysis
    results_df = results.to_pandas()

    # Extract aggregate scores from DataFrame
    scores = {}
    for metric_name in selected_metric_names:
        col_name = RAGAS_METRIC_COLUMN_MAP.get(metric_name, metric_name)
        if col_name in results_df.columns:
            scores[metric_name] = float(results_df[col_name].mean())

    # Populate per-question scores into traces
    for i, trace in enumerate(question_traces):
        if i < len(results_df):
            for metric_name in selected_metric_names:
                col_name = RAGAS_METRIC_COLUMN_MAP.get(metric_name, metric_name)
                if col_name in results_df.columns:
                    value = results_df.iloc[i].get(col_name)
                    if _is_valid_score(value):
                        trace.scores[metric_name] = float(value)

    # Compute difficulty breakdown
    difficulty_breakdown = compute_difficulty_breakdown(
        results_df=results_df,
        test_questions=test_questions,
        metrics=selected_metric_names,
    )

    # Build and save evaluation trace
    saved_trace_path = None
    if save_trace:
        evaluation_trace = EvaluationTrace(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            questions=question_traces,
            aggregate_scores=scores,
            difficulty_breakdown=difficulty_breakdown,
            ragas_metrics_used=selected_metric_names,
            evaluation_model=evaluation_model,
        )

        # Determine trace path
        if trace_path:
            saved_trace_path = trace_path
        else:
            saved_trace_path = EVAL_TRACES_DIR / f"trace_{run_id}.json"

        evaluation_trace.save(saved_trace_path)
        logger.info(f"Trace saved to: {saved_trace_path}")

    return {
        "scores": scores,
        "results": results_df,
        "samples": samples,
        "difficulty_breakdown": difficulty_breakdown,
        "trace_path": saved_trace_path,
    }
