"""Vector database integration for RAG1-Mini.

Provides:
- Weaviate client wrapper for collection management
- Batch upload functionality for embeddings
"""

from src.vector_db.weaviate_client import (
    get_client,
    create_collection,
    delete_collection,
    upload_embeddings,
    get_collection_count,
)

__all__ = [
    "get_client",
    "create_collection",
    "delete_collection",
    "upload_embeddings",
    "get_collection_count",
]
