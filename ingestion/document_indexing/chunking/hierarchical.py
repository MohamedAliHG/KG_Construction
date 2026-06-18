"""Hierarchical Docling chunker."""

from __future__ import annotations

from ingestion.document_indexing.chunking.base import BaseChunker
from ingestion.document_indexing.chunking.docling_utils import chunk_to_document_chunk
from ingestion.document_indexing.chunking.registry import register_chunker
from ingestion.document_indexing.types import DocumentChunk, ParsedDocument

from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker as DoclingHierarchicalChunker,
)


@register_chunker("hierarchical")
class HierarchicalChunker(BaseChunker):
    def __init__(self, *, merge_list_items: bool) -> None:
        super().__init__(config=None)
        self.merge_list_items = merge_list_items
        self._chunker = DoclingHierarchicalChunker(merge_list_items=self.merge_list_items)

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for idx, chunk in enumerate(self._chunker.chunk(document.dl_doc)):
            text = getattr(chunk, "text", None) or getattr(chunk, "content", None)
            if not text:
                continue
            chunks.append(
                chunk_to_document_chunk(
                    document=document,
                    strategy=self.name,
                    chunk_index=idx,
                    chunk=chunk,
                    text=str(text).strip(),
                )
            )
        return chunks
