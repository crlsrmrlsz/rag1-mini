"""RAGAS evaluation for RAG1-Mini.

Provides:
- OpenRouter chat integration for answer generation
- RAGAS evaluation with LangChain wrapper
- Retrieval and generation functions for the evaluation pipeline
"""

from typing import List, Dict, Any, Optional, Callable

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    FactualCorrectness,
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
)
from src.rag_pipeline.indexing import get_client, query_hybrid
from src.rag_pipeline.retrieval.reranking import rerank
from src.shared.files import setup_logging
from src.shared.openrouter_client import call_simple_prompt

logger = setup_logging(__name__)


# Default model for answer generation (can be overridden)
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"


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


# ============================================================================
# RETRIEVAL AND GENERATION
# ============================================================================


def retrieve_contexts(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
) -> List[str]:
    """
    Retrieve relevant contexts from Weaviate for a question.

    This function implements two-stage retrieval:
    1. Hybrid search retrieves an initial candidate set
    2. Cross-encoder reranking refines the results

    Args:
        question: The user's question.
        top_k: Number of chunks to return after reranking.
        collection_name: Override collection name.
        use_reranking: If True, apply cross-encoder reranking.
                       Retrieves 50 candidates and reranks to top_k.
                       If False, directly returns top_k from hybrid search.
        alpha: Hybrid search balance (0.0=keyword, 0.5=balanced, 1.0=vector).

    Returns:
        List of context strings from retrieved chunks.

    Technical Notes:
        - Hybrid search combines vector similarity with BM25 keyword matching
        - alpha=0.0 is pure BM25, alpha=1.0 is pure vector, 0.5 is balanced
        - Cross-encoder processes [query, document] pairs for deeper matching
        - Reranking improves precision by 20-35% (research evidence)
    """
    collection_name = collection_name or get_collection_name()
    client = get_client()

    try:
        # Determine initial retrieval size
        # If reranking, get more candidates for the reranker to work with
        initial_k = 50 if use_reranking else top_k

        results = query_hybrid(
            client=client,
            query_text=question,
            top_k=initial_k,
            alpha=alpha,
            collection_name=collection_name,
        )

        # Apply cross-encoder reranking if enabled
        if use_reranking and results:
            results = rerank(question, results, top_k=top_k)

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
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on test questions.

    This function:
    1. Optionally preprocesses each question (step-back, multi-query, etc.)
    2. Retrieves contexts for each question (with optional cross-encoder reranking)
    3. Generates answers using the RAG pipeline
    4. Evaluates using RAGAS metrics

    Args:
        test_questions: List of dicts with 'question' and optionally 'reference' keys.
        metrics: Which metrics to compute. Options:
            - "faithfulness": Is the answer grounded in context?
            - "relevancy": Does the answer address the question?
            - "context_precision": Are retrieved chunks relevant?
            - "context_recall": Did retrieval capture needed info? (requires reference)
            - "factual_correctness": Is the answer correct? (requires reference)
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

    Returns:
        Dict with:
            - "scores": Dict of metric_name -> average score
            - "results": Per-question results DataFrame
            - "samples": List of evaluation samples
    """
    # Default metrics - includes reference-based metrics since all questions have ground truth
    if metrics is None:
        metrics = [
            "faithfulness",
            "relevancy",
            "context_precision",
            "context_recall",
            "factual_correctness",
        ]

    logger.info(f"Starting evaluation with {len(test_questions)} questions")
    logger.info(f"Metrics: {metrics}")

    # Build evaluation samples
    samples = []
    for i, q in enumerate(test_questions):
        question = q["question"]
        reference = q.get("reference")

        logger.info(f"Processing question {i + 1}/{len(test_questions)}: {question[:50]}...")

        # Apply preprocessing if enabled
        search_query = question
        if preprocessing_strategy != "none":
            from src.rag_pipeline.retrieval.preprocessing import preprocess_query
            try:
                preprocessed = preprocess_query(
                    query=question,
                    strategy=preprocessing_strategy,
                    model=preprocessing_model,
                )
                search_query = preprocessed.search_query
                logger.info(f"  Preprocessed ({preprocessing_strategy}): {search_query[:50]}...")
            except Exception as e:
                logger.warning(f"  Preprocessing failed: {e}. Using original query.")
                search_query = question

        # Retrieve contexts (with optional reranking for improved precision)
        contexts = retrieve_contexts(
            question=search_query,  # Use preprocessed query for retrieval
            top_k=top_k,
            collection_name=collection_name,
            use_reranking=use_reranking,
            alpha=alpha,
        )
        logger.info(f"  Retrieved {len(contexts)} contexts")

        # Generate answer
        answer = generate_answer(
            question=question,
            contexts=contexts,
            model=generation_model,
        )
        logger.info(f"  Generated answer: {answer[:100]}...")

        sample = {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
        }

        # Add reference if available (needed for some metrics)
        if reference:
            sample["reference"] = reference

        samples.append(sample)

    # Create RAGAS dataset
    dataset = EvaluationDataset.from_list(samples)

    # Map metric names to objects
    metric_map = {
        "faithfulness": Faithfulness(),
        "relevancy": ResponseRelevancy(),
        "context_precision": LLMContextPrecisionWithoutReference(),
        "context_recall": LLMContextRecall(),
        "factual_correctness": FactualCorrectness(),
    }

    # Validate metrics
    selected_metrics = []
    for m in metrics:
        if m not in metric_map:
            logger.warning(f"Unknown metric: {m}, skipping")
            continue

        # Check if metric requires reference
        if m in ["context_recall", "factual_correctness"]:
            has_references = all(q.get("reference") for q in test_questions)
            if not has_references:
                logger.warning(f"Metric {m} requires reference answers, skipping")
                continue

        selected_metrics.append(metric_map[m])

    if not selected_metrics:
        raise ValueError("No valid metrics selected")

    logger.info(f"Running RAGAS evaluation with {len(selected_metrics)} metrics...")

    # Create evaluator LLM and embeddings
    evaluator_llm = create_evaluator_llm(model=evaluation_model)
    evaluator_embeddings = create_evaluator_embeddings()

    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    logger.info("Evaluation complete")

    # Convert to DataFrame for analysis
    results_df = results.to_pandas()

    # Map our metric names to RAGAS column names
    metric_column_map = {
        "faithfulness": "faithfulness",
        "relevancy": "answer_relevancy",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
        "factual_correctness": "factual_correctness",
    }

    # Extract aggregate scores from DataFrame
    scores = {}
    for metric_name in metrics:
        col_name = metric_column_map.get(metric_name, metric_name)
        if col_name in results_df.columns:
            scores[metric_name] = float(results_df[col_name].mean())

    return {
        "scores": scores,
        "results": results_df,
        "samples": samples,
    }
