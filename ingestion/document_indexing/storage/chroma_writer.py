"""Local persistent ChromaDB writer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import settings
from ingestion.document_indexing.types import EmbeddedChunk

logger = logging.getLogger(__name__)


class ChromaWriter:
    """Write embedded chunks to local persistent ChromaDB."""

    def __init__(
        self,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        namespace: str | None = None,
        reset_collection: bool | None = None,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.persist_directory = persist_directory or settings.chroma_path
        self.collection_name = collection_name or settings.chroma_collection
        self.namespace = namespace or settings.chroma_namespace or "default"
        self.reset_collection = (
            settings.document_chroma_reset_collection
            if reset_collection is None
            else reset_collection
        )

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        if self.reset_collection:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info("Dropped Chroma collection '%s'", self.collection_name)
            except Exception:
                logger.info("Collection '%s' did not exist", self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def write(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        if not embedded_chunks:
            return 0

        ids = [item.chunk.chunk_id for item in embedded_chunks]
        documents = [item.chunk.text for item in embedded_chunks]
        embeddings = [item.embedding for item in embedded_chunks]
        metadatas = [
            _sanitize_metadata({**item.chunk.metadata, "namespace": self.namespace})
            for item in embedded_chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def count(self) -> int:
        return self.collection.count()


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    """Convert metadata to Chroma-compatible scalar values."""
    sanitized: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, Path):
            sanitized[key] = str(value)
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False, default=str)
    return sanitized
