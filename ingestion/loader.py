from __future__ import annotations

import logging
from typing import Iterator, Sequence

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
    pages: Sequence[int] | None = None,
) -> Iterator[list[Document]]:
    collection_name = collection_name or settings.chroma_collection
    batch_size = batch_size or settings.batch_size

    client = _chroma_client()
    collection = client.get_collection(collection_name)
    page_numbers = _normalize_pages(pages)
    where = _build_where_filter(namespace=namespace, pages=page_numbers)

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
        filter_parts = []
        if namespace is not None:
            filter_parts.append(f"namespace='{namespace}'")
        if page_numbers:
            filter_parts.append(f"pages={page_numbers}")
        logger.info(
            "ChromaDB collection '%s' — %d chunks found at '%s' with %s",
            collection_name,
            total,
            settings.chroma_path,
            ", ".join(filter_parts),
        )

    offset = 0
    while offset < total:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
            where=where,
        )

        docs = []
        ids = _safe_result_list(result.get("ids"))
        documents = _safe_result_list(result.get("documents"))
        metadatas = _safe_result_list(result.get("metadatas"))
        embeddings = _safe_result_list(result.get("embeddings"))

        for idx, (chunk_id, text, meta) in enumerate(zip(ids, documents, metadatas)):
            metadata = dict(meta or {})
            if chunk_id and not metadata.get("id"):
                metadata["id"] = chunk_id
            if chunk_id and not metadata.get("chunk_id"):
                metadata["chunk_id"] = chunk_id

            embedding = _coerce_embedding(embeddings[idx] if idx < len(embeddings) else None)
            if embedding is not None:
                metadata["embedding"] = embedding
                metadata["embedding_model"] = (
                    metadata.get("embedding_model") or settings.document_embedding_model_name
                )
                metadata["embedding_dim"] = len(embedding)

            docs.append(Document(page_content=text, metadata=metadata))

        logger.debug("Loaded batch offset=%d, size=%d", offset, len(docs))
        yield docs
        offset += batch_size


def load_all_chunks(
    collection_name: str | None = None,
    namespace: str | None = None,
    pages: Sequence[int] | None = None,
) -> list[Document]:
    return [
        doc
        for batch in load_chunks(
            collection_name=collection_name,
            batch_size=500,
            namespace=namespace,
            pages=pages,
        )
        for doc in batch
    ]


def _build_where_filter(
    *,
    namespace: str | None,
    pages: tuple[int, ...],
) -> dict | None:
    filters = []
    if namespace is not None:
        filters.append({"namespace": namespace})
    if pages:
        filters.append({"page_no": {"$in": list(pages)}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _normalize_pages(pages: Sequence[int] | None) -> tuple[int, ...]:
    if not pages:
        return ()
    return tuple(sorted({int(page) for page in pages}))


def _coerce_embedding(value) -> list[float] | None:
    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            logger.debug("Skipping non-numeric embedding value: %r", value)
            return None

    try:
        return [float(item) for item in list(value)]
    except TypeError:
        logger.debug("Skipping unsupported embedding value: %r", value)
        return None


def _safe_result_list(value):
    if value is None:
        return []
    return list(value)
