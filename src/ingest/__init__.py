"""Ingest package for RAG1-Mini.

Contains modules for chunking and embedding text.
"""

from .naive_chunker import run_section_chunking, create_chunks_from_paragraphs
from .embed_texts import embed_texts

__all__ = ['run_section_chunking', 'create_chunks_from_paragraphs', 'embed_texts']
