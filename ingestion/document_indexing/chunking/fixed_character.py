"""Markdown-friendly recursive character chunker."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from ingestion.document_indexing.chunking.base import BaseChunker
from ingestion.document_indexing.chunking.registry import register_chunker
from ingestion.document_indexing.types import DocumentChunk, ParsedDocument


@register_chunker("fixed_character")
class FixedCharacterChunker(BaseChunker):
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        super().__init__(config=None)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.markdown.strip()
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        for idx, piece_text in enumerate(self._splitter.split_text(text)):
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
