"""Vector database integration for RAG1-Mini.

Provides:
- Weaviate client wrapper for collection management
- Batch upload functionality for embeddings
- Query functions for similarity and hybrid search
"""

from src.vector_db.weaviate_client import (
    get_client,
    create_collection,
    delete_collection,
    upload_embeddings,
    get_collection_count,
)

from src.vector_db.weaviate_query import (
    SearchResult,
    query_similar,
    query_hybrid,
    list_available_books,
)

__all__ = [
    # Client management
    "get_client",
    "create_collection",
    "delete_collection",
    "upload_embeddings",
    "get_collection_count",
    # Query functions
    "SearchResult",
    "query_similar",
    "query_hybrid",
    "list_available_books",
]
