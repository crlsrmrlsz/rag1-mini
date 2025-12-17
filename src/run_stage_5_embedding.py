"""
Stage 5: Embed final text chunks for RAG.

This stage:
- Loads section-level chunks from Stage 4
- Calls embedding API (OpenAI-compatible)
- Saves embeddings to disk (no vector DB yet)

Design goals:
- Deterministic
- Restartable
- Transparent
"""

import json
from pathlib import Path
from typing import List, Dict

from src.config import (
    DIR_FINAL_CHUNKS,
    PROJECT_ROOT,
    TOKENIZER_MODEL,
    DIR_EMBEDDINGS,
    EMBEDDING_MODEL,
    MAX_BATCH_TOKENS,
    MAX_RETRIES 
)

from src.utils.file_utils import setup_logging, get_file_list
from src.utils.tokens import count_tokens
from src.ingest.embed_texts import embed_texts

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Where embeddings will be stored

DIR_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

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


def embed_book(file_path: Path):
    """
    Embed all chunks for a single book.
    """
    logger.info(f"Embedding book: {file_path.stem}")

    chunks = load_chunks(file_path)
    batches = batch_chunks_by_token_limit(chunks)

    embedded_chunks = []

    for batch_idx, batch in enumerate(batches):
        texts = [c["text"] for c in batch]

        logger.info(
            f"  → Batch {batch_idx + 1}/{len(batches)} "
            f"({sum(c['token_count'] for c in batch)} tokens)"
        )

        embeddings = embed_texts(texts)

        for chunk, vector in zip(batch, embeddings):
            embedded_chunks.append({
                **chunk,
                "embedding": vector,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": len(vector)
            })

    # Save per-book embedding file
    output_path = DIR_EMBEDDINGS / f"{file_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "book_id": file_path.stem,
            "embedding_model": EMBEDDING_MODEL,
            "chunks": embedded_chunks
        }, f, ensure_ascii=False, indent=2)

    logger.info(
        f"  ✓ Saved {len(embedded_chunks)} embeddings → {output_path}"
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting Stage 5: Embedding")

    section_dir = DIR_FINAL_CHUNKS / "section"
    files = list(section_dir.glob("*.json"))

    if not files:
        logger.warning("No section chunks found. Run Stage 4 first.")
        return

    logger.info(f"Found {len(files)} books to embed.")

    for file_path in files:
        try:
            embed_book(file_path)
        except Exception as e:
            logger.error(f"Failed embedding {file_path.name}: {e}")
            raise

    logger.info("Stage 5 complete.")


if __name__ == "__main__":
    main()
