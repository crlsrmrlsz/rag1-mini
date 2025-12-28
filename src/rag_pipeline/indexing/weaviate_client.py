"""Weaviate client wrapper for RAGLab.

Provides functions for:
- Connecting to local Weaviate instance
- Creating/deleting collections with appropriate schema
- Batch uploading embeddings with metadata

Uses Weaviate Python client v4 (requires gRPC).
"""

from typing import List, Dict, Any
import uuid

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

from src.config import (
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_BATCH_SIZE,
)


def get_client() -> weaviate.WeaviateClient:
    """
    Create and return a Weaviate client connected to local instance.

    Returns:
        Connected WeaviateClient instance.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If connection fails.
    """
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
    return client


def create_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> None:
    """
    Create a new collection with the RAG chunk schema.

    Args:
        client: Connected Weaviate client.
        collection_name: Name for the new collection.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If collection creation fails.
    """
    client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="book_id", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="context", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="token_count", data_type=DataType.INT),
            Property(name="chunking_strategy", data_type=DataType.TEXT),
            Property(name="embedding_model", data_type=DataType.TEXT),
        ],
    )


def create_raptor_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> None:
    """
    Create a collection with extended schema for RAPTOR trees.

    Extends the base RAG chunk schema with tree-specific properties:
    - tree_level: Depth in tree (0=leaf, 1+=summary)
    - is_summary: Quick filter for summary nodes
    - parent_ids: Parent chunk IDs (for tree traversal)
    - child_ids: Child chunk IDs (for tree traversal)
    - cluster_id: Cluster identifier at this level
    - source_chunk_ids: (Summaries) Original leaf chunks in subtree

    Args:
        client: Connected Weaviate client.
        collection_name: Name for the new collection.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If collection creation fails.
    """
    client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            # Base chunk properties (same as create_collection)
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="book_id", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="context", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="token_count", data_type=DataType.INT),
            Property(name="chunking_strategy", data_type=DataType.TEXT),
            Property(name="embedding_model", data_type=DataType.TEXT),
            # RAPTOR-specific tree properties
            Property(name="tree_level", data_type=DataType.INT),
            Property(name="is_summary", data_type=DataType.BOOL),
            Property(name="parent_ids", data_type=DataType.TEXT_ARRAY),
            Property(name="child_ids", data_type=DataType.TEXT_ARRAY),
            Property(name="cluster_id", data_type=DataType.TEXT),
            Property(name="source_chunk_ids", data_type=DataType.TEXT_ARRAY),
        ],
    )


def delete_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> bool:
    """
    Delete a collection if it exists.

    Args:
        client: Connected Weaviate client.
        collection_name: Name of collection to delete.

    Returns:
        True if collection was deleted, False if it did not exist.
    """
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        return True
    return False


def _generate_uuid_from_chunk_id(chunk_id: str) -> str:
    """
    Generate deterministic UUID from chunk_id for idempotent uploads.

    Args:
        chunk_id: Unique chunk identifier.

    Returns:
        UUID string derived from chunk_id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def upload_embeddings(
    client: weaviate.WeaviateClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    batch_size: int = WEAVIATE_BATCH_SIZE,
    is_raptor: bool = False,
) -> int:
    """
    Upload embedded chunks to a Weaviate collection.

    Handles both regular chunks and RAPTOR tree nodes. For RAPTOR nodes,
    includes additional tree properties (tree_level, parent_ids, etc.).

    Args:
        client: Connected Weaviate client.
        collection_name: Target collection name.
        chunks: List of chunk dicts with 'embedding' and metadata fields.
        batch_size: Number of objects per batch (default from config).
        is_raptor: If True, include RAPTOR tree properties.

    Returns:
        Number of successfully uploaded objects.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If batch upload fails.
    """
    collection = client.collections.get(collection_name)
    uploaded_count = 0

    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for chunk in chunks:
            # Base properties (all strategies)
            properties = {
                "chunk_id": chunk["chunk_id"],
                "book_id": chunk["book_id"],
                "section": chunk.get("section", ""),
                "context": chunk.get("context", ""),
                "text": chunk["text"],
                "token_count": chunk.get("token_count", 0),
                "chunking_strategy": chunk.get("chunking_strategy", ""),
                "embedding_model": chunk.get("embedding_model", ""),
            }

            # Add RAPTOR tree properties if applicable
            if is_raptor:
                properties.update({
                    "tree_level": chunk.get("tree_level", 0),
                    "is_summary": chunk.get("is_summary", False),
                    "parent_ids": chunk.get("parent_ids", []),
                    "child_ids": chunk.get("child_ids", []),
                    "cluster_id": chunk.get("cluster_id", ""),
                    "source_chunk_ids": chunk.get("source_chunk_ids", []),
                })

            batch.add_object(
                properties=properties,
                vector=chunk["embedding"],
                uuid=_generate_uuid_from_chunk_id(chunk["chunk_id"]),
            )
            uploaded_count += 1

    return uploaded_count


def get_collection_count(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> int:
    """
    Get the number of objects in a collection.

    Args:
        client: Connected Weaviate client.
        collection_name: Name of collection to count.

    Returns:
        Number of objects in the collection.
    """
    collection = client.collections.get(collection_name)
    result = collection.aggregate.over_all(total_count=True)
    return result.total_count
