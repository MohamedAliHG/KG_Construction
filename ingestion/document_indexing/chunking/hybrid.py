"""Hybrid Docling chunker with doc structure plus token limits."""

from __future__ import annotations

from ingestion.document_indexing.chunking.base import BaseChunker
from ingestion.document_indexing.chunking.docling_utils import chunk_to_document_chunk
from ingestion.document_indexing.chunking.registry import register_chunker
from ingestion.document_indexing.types import DocumentChunk, ParsedDocument


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

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        dl_chunks = getattr(document.dl_doc, "chunks", None) or []

        if dl_chunks:
            for idx, chunk in enumerate(dl_chunks):
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

        if chunks:
            return chunks

        text = document.markdown.strip()
        if not text:
            return []

        words = text.split()
        step = max(1, self.max_tokens)
        for idx, start in enumerate(range(0, len(words), step)):
            piece_text = " ".join(words[start : start + step]).strip()
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
