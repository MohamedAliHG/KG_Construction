"""Common chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ingestion.document_indexing.types import DocumentChunk, ParsedDocument


class BaseChunker(ABC):
    """Interface implemented by every chunking strategy."""

    name: str

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Split a parsed document into chunks."""

    def _metadata(self, document: ParsedDocument, chunk_index: int) -> dict:
        return {
            **document.metadata,
            "chunking_strategy": self.name,
            "chunk_index": chunk_index,
        }
