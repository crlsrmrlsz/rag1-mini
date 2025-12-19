"""Weaviate search service for RAG UI.

Provides semantic search with optional book filtering.
This module wraps the vector_db query functions for use in the Streamlit UI.
"""

from typing import List, Dict, Any, Optional

from src.config import get_collection_name, DEFAULT_TOP_K
from src.vector_db import get_client, query_similar, query_hybrid, SearchResult


def search_chunks(
    query: str,
    book_filter: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    search_type: str = "vector",
    alpha: float = 0.5,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search Weaviate for relevant chunks.

    This is the main search function for the UI. It connects to Weaviate,
    executes the search, and returns results as dictionaries.

    Args:
        query: User's search query.
        book_filter: List of book_ids to include (None = all books).
        top_k: Number of results to return.
        search_type: Either "vector" (semantic) or "hybrid" (vector + keyword).
        alpha: For hybrid search, balance between vector (1.0) and keyword (0.0).
        collection_name: Override collection (for future multi-collection).

    Returns:
        List of chunk dictionaries with text, metadata, and similarity score.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If Weaviate is not running.
    """
    collection_name = collection_name or get_collection_name()

    # Connect to Weaviate
    client = get_client()

    try:
        # Execute search based on type
        if search_type == "hybrid":
            results: List[SearchResult] = query_hybrid(
                client=client,
                query_text=query,
                top_k=top_k,
                alpha=alpha,
                book_ids=book_filter,
                collection_name=collection_name,
            )
        else:
            results: List[SearchResult] = query_similar(
                client=client,
                query_text=query,
                top_k=top_k,
                book_ids=book_filter,
                collection_name=collection_name,
            )

        # Convert SearchResult objects to dicts for Streamlit
        return [
            {
                "chunk_id": r.chunk_id,
                "book_id": r.book_id,
                "section": r.section,
                "context": r.context,
                "text": r.text,
                "token_count": r.token_count,
                "similarity": r.score,
            }
            for r in results
        ]

    finally:
        client.close()


def list_collections() -> List[str]:
    """
    List all available RAG collections in Weaviate.

    Returns:
        List of collection names starting with 'RAG_'.
    """
    client = get_client()

    try:
        all_collections = client.collections.list_all()
        return sorted([name for name in all_collections.keys() if name.startswith("RAG_")])
    finally:
        client.close()
