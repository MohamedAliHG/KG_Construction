from __future__ import annotations

import logging
from typing import Iterator

import chromadb
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)


def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_path)


def load_chunks(
    collection_name: str | None = None,
    batch_size: int | None = None,
    namespace: str | None = None,
) -> Iterator[list[Document]]:
    collection_name = collection_name or settings.chroma_collection
    batch_size = batch_size or settings.batch_size

    client = _chroma_client()
    collection = client.get_collection(collection_name)
    where = {"namespace": namespace} if namespace is not None else None

    if where is None:
        total = collection.count()
        logger.info(
            "ChromaDB collection '%s' — %d chunks found at '%s'",
            collection_name,
            total,
            settings.chroma_path,
        )
    else:
        total = len(collection.get(where=where, include=["metadatas"])["ids"])
        logger.info(
            "ChromaDB collection '%s' — %d chunks found at '%s' with namespace='%s'",
            collection_name,
            total,
            settings.chroma_path,
            namespace,
        )

    offset = 0
    while offset < total:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
            where=where,
        )

        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(result["documents"], result["metadatas"])
        ]

        logger.debug("Loaded batch offset=%d, size=%d", offset, len(docs))
        yield docs
        offset += batch_size


def load_all_chunks(
    collection_name: str | None = None,
    namespace: str | None = None,
) -> list[Document]:
    return [
        doc
        for batch in load_chunks(
            collection_name=collection_name,
            batch_size=500,
            namespace=namespace,
        )
        for doc in batch
    ]
