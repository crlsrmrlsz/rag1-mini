"""
Stage 6: Upload embeddings to Weaviate vector database.

This stage:
- Loads embedded chunks from Stage 5
- Creates a new Weaviate collection (deletes if exists)
- Batch uploads all chunks with their embeddings

Design goals:
- Clean slate: always recreates collection
- Deterministic: same input produces same UUIDs
- Transparent: logs progress per book
"""

import json
from pathlib import Path
from typing import List, Dict

from src.config import (
    DIR_EMBEDDINGS,
    get_collection_name,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
)

from src.shared.file_utils import setup_logging
from src.rag_pipeline.indexing import (
    get_client,
    create_collection,
    delete_collection,
    upload_embeddings,
    get_collection_count,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

logger = setup_logging("Stage6_Weaviate")


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def load_embedding_file(file_path: Path) -> List[Dict]:
    """
    Load embedded chunks from a JSON file.

    Args:
        file_path: Path to the embedding JSON file.

    Returns:
        List of chunk dictionaries with embeddings.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def upload_book(client, collection_name: str, file_path: Path) -> int:
    """
    Upload all chunks from a single book to Weaviate.

    Args:
        client: Connected Weaviate client.
        collection_name: Target collection name.
        file_path: Path to the embedding JSON file.

    Returns:
        Number of chunks uploaded.
    """
    logger.info(f"Uploading: {file_path.stem}")

    chunks = load_embedding_file(file_path)

    if not chunks:
        logger.warning(f"  No chunks found in {file_path.name}")
        return 0

    count = upload_embeddings(client, collection_name, chunks)
    logger.info(f"  Uploaded {count} chunks")

    return count


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """Main entry point for Stage 6."""
    collection_name = get_collection_name()

    logger.info("Starting Stage 6: Weaviate Upload")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Weaviate: http://{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}")

    # Get embedding files
    files = list(DIR_EMBEDDINGS.glob("*.json"))

    if not files:
        logger.warning("No embedding files found. Run Stage 5 first.")
        return

    logger.info(f"Found {len(files)} books to upload")

    # Connect to Weaviate
    client = get_client()

    try:
        # Clean slate: delete existing collection
        if delete_collection(client, collection_name):
            logger.info(f"Deleted existing collection: {collection_name}")

        # Create fresh collection
        create_collection(client, collection_name)
        logger.info(f"Created collection: {collection_name}")

        # Upload each book
        total_chunks = 0
        for file_path in sorted(files):
            try:
                count = upload_book(client, collection_name, file_path)
                total_chunks += count
            except Exception as e:
                logger.error(f"Failed uploading {file_path.name}: {e}")
                raise

        # Verify final count
        final_count = get_collection_count(client, collection_name)
        logger.info(f"Stage 6 complete: {final_count} chunks in {collection_name}")

        if final_count != total_chunks:
            logger.warning(
                f"Count mismatch: uploaded {total_chunks}, found {final_count}"
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
