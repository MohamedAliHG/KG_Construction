from __future__ import annotations

import logging
from typing import Iterator

import chromadb
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)


def _chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client pointed at the configured path."""
    return chromadb.PersistentClient(path=settings.chroma_path)


def load_chunks(
    collection_name: str | None = None,
    batch_size: int | None = None,
) -> Iterator[list[Document]]:
    
    collection_name = collection_name or settings.chroma_collection
    batch_size = batch_size or settings.batch_size

    client = _chroma_client()
    collection = client.get_collection(collection_name)

    total = collection.count()
    logger.info(
        "ChromaDB collection '%s' — %d chunks found at '%s'",
        collection_name,
        total,
        settings.chroma_path,
    )

    offset = 0
    while offset < total:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(result["documents"], result["metadatas"])
        ]

        logger.debug("Loaded batch offset=%d, size=%d", offset, len(docs))
        yield docs
        offset += batch_size


def load_all_chunks(collection_name: str | None = None) -> list[Document]:
    """Convenience wrapper — loads every chunk into a single flat list."""
    return [
        doc
        for batch in load_chunks(collection_name=collection_name, batch_size=500)
        for doc in batch
    ]