"""
Stage 5: Embed final text chunks for RAG.

This stage:
- Loads chunks from Stage 4 (strategy-specific directory)
- Calls embedding API (OpenAI-compatible)
- Saves embeddings to disk (no vector DB yet)

Design goals:
- Deterministic
- Restartable
- Transparent

Usage:
    python -m src.stages.run_stage_5_embedding                    # Default: section
    python -m src.stages.run_stage_5_embedding --strategy semantic  # Semantic chunks
    python -m src.stages.run_stage_5_embedding --strategy semantic --threshold 0.6
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from src.config import (
    DIR_FINAL_CHUNKS,
    PROJECT_ROOT,
    TOKENIZER_MODEL,
    EMBEDDING_MODEL,
    MAX_BATCH_TOKENS,
    MAX_RETRIES,
    DEFAULT_CHUNKING_STRATEGY,
    SEMANTIC_SIMILARITY_THRESHOLD,
    get_semantic_folder_name,
    get_embedding_folder_path,
)

from src.shared.files import setup_logging, get_file_list, OverwriteContext, parse_overwrite_arg
from src.shared.tokens import count_tokens
from src.rag_pipeline.embedding.embedder import embed_texts
from src.rag_pipeline.chunking.strategies import list_strategies

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

logger = setup_logging("Stage5_Embedding")

# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def load_chunks(file_path: Path) -> List[Dict]:
    """Load chunk list from a JSON file."""
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def batch_chunks_by_token_limit(chunks: List[Dict]) -> List[List[Dict]]:
    """
    Group chunks into batches that stay under MAX_BATCH_TOKENS.

    This prevents API failures and rate-limit issues.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        tokens = chunk["token_count"]

        # Single chunk too large (should not happen, but be safe)
        if tokens > MAX_BATCH_TOKENS:
            logger.warning(
                f"Chunk {chunk['chunk_id']} exceeds batch limit "
                f"({tokens} tokens). Embedding alone."
            )
            batches.append([chunk])
            continue

        if current_tokens + tokens > MAX_BATCH_TOKENS:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(chunk)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def embed_book(file_path: Path, output_dir: Path):
    """
    Embed all chunks for a single book.

    Args:
        file_path: Path to the input chunk JSON file.
        output_dir: Directory where embedding file will be saved.

    Note:
        Preserves all chunk fields from the input (including strategy-specific
        fields like 'original_text' and 'contextual_snippet' for contextual
        chunking). Adds embedding fields: 'embedding', 'embedding_model',
        'embedding_dim'.
    """
    logger.info(f"Embedding book: {file_path.stem}")

    chunks = load_chunks(file_path)
    batches = batch_chunks_by_token_limit(chunks)

    embedded_chunks = []

    for batch_idx, batch in enumerate(batches):
        texts = [c["text"] for c in batch]

        logger.info(
            f"  Batch {batch_idx + 1}/{len(batches)} "
            f"({sum(c['token_count'] for c in batch)} tokens)"
        )

        embeddings = embed_texts(texts)

        for chunk, vector in zip(batch, embeddings):
            # Preserve all existing chunk fields (strategy-specific included)
            embedded_chunks.append({
                **chunk,
                "embedding": vector,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": len(vector)
            })

    # Save per-book embedding file
    output_path = output_dir / f"{file_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "book_id": file_path.stem,
            "embedding_model": EMBEDDING_MODEL,
            "chunks": embedded_chunks
        }, f, ensure_ascii=False, indent=2)

    logger.info(
        f"  Saved {len(embedded_chunks)} embeddings to {output_path}"
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Embed chunks from specified chunking strategy"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=list_strategies(),
        help=f"Chunking strategy to embed (default: {DEFAULT_CHUNKING_STRATEGY})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Semantic similarity threshold (for finding correct input folder). "
            f"Only used with semantic strategy. (default: {SEMANTIC_SIMILARITY_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        choices=["prompt", "skip", "all"],
        default="prompt",
        help="Overwrite behavior: prompt (default), skip, all",
    )
    args = parser.parse_args()

    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    # Determine strategy key for paths (semantic uses threshold-based naming)
    if args.strategy == "semantic":
        threshold = args.threshold if args.threshold is not None else SEMANTIC_SIMILARITY_THRESHOLD
        strategy_key = get_semantic_folder_name(threshold)
    else:
        strategy_key = args.strategy

    logger.info(f"Starting Stage 5: Embedding (strategy: {strategy_key})")

    # Determine input directory (chunks from Stage 4)
    strategy_dir = DIR_FINAL_CHUNKS / strategy_key
    files = list(strategy_dir.glob("*.json")) if strategy_dir.exists() else []

    if not files:
        logger.warning(
            f"No chunks found in {strategy_dir}. "
            f"Run Stage 4 with --strategy {args.strategy} first."
        )
        return

    # Determine output directory (strategy-scoped embeddings)
    output_dir = get_embedding_folder_path(strategy_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input:  {strategy_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Found {len(files)} books to embed")

    success_count = 0
    skipped_count = 0
    for file_path in files:
        # Check overwrite decision (use strategy-scoped output path)
        output_path = output_dir / f"{file_path.stem}.json"
        if not overwrite_context.should_overwrite(output_path, logger):
            skipped_count += 1
            continue

        try:
            embed_book(file_path, output_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed embedding {file_path.name}: {e}")
            raise

    logger.info(f"Stage 5 complete ({strategy_key}). {success_count} embedded, {skipped_count} skipped.")


if __name__ == "__main__":
    main()
