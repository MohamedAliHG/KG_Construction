"""Hybrid Docling chunker with document structure plus token limits."""

from __future__ import annotations

from ingestion.document_indexing.chunking.base import BaseChunker
from ingestion.document_indexing.chunking.docling_utils import (
    build_huggingface_tokenizer,
    build_markdown_serializer_provider,
    chunk_to_document_chunk,
)
from ingestion.document_indexing.chunking.registry import register_chunker
from ingestion.document_indexing.types import DocumentChunk, ParsedDocument

from docling.chunking import HybridChunker as DoclingHybridChunker


@register_chunker("hybrid")
class HybridChunker(BaseChunker):
    def __init__(
        self,
        *,
        tokenizer: str,
        max_tokens: int,
        merge_peers: bool,
        image_mode: str,
        image_placeholder: str,
        mark_annotations: bool,
        include_annotations: bool,
    ) -> None:
        super().__init__(config=None)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.image_mode = image_mode
        self.image_placeholder = image_placeholder
        self.mark_annotations = mark_annotations
        self.include_annotations = include_annotations
        self._chunker = DoclingHybridChunker(
            tokenizer=build_huggingface_tokenizer(self.tokenizer, self.max_tokens),
            serializer_provider=build_markdown_serializer_provider(
                image_mode=self.image_mode,
                image_placeholder=self.image_placeholder,
                mark_annotations=self.mark_annotations,
                include_annotations=self.include_annotations,
            ),
            merge_peers=self.merge_peers,
        )

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
