"""Character-count based chunker."""

from __future__ import annotations

from ingestion.document_indexing.chunking.base import BaseChunker
from ingestion.document_indexing.chunking.registry import register_chunker
from ingestion.document_indexing.types import DocumentChunk, ParsedDocument


@register_chunker("fixed_character")
class FixedCharacterChunker(BaseChunker):
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        super().__init__(config=None)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.markdown.strip()
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for idx, start in enumerate(range(0, len(text), step)):
            piece_text = text[start : start + self.chunk_size].strip()
            if not piece_text:
                break
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.doc_id}_{self.name}_{idx}",
                    text=piece_text,
                    metadata=self._metadata(document, idx),
                )
            )
        return chunks
