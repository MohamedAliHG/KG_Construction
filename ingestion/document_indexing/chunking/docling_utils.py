"""Helpers for Docling-native chunkers."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from ingestion.document_indexing.types import DocumentChunk, ParsedDocument


def chunk_to_document_chunk(
    *,
    document: ParsedDocument,
    strategy: str,
    chunk_index: int,
    chunk: Any,
    text: str,
) -> DocumentChunk:
    metadata = {
        **document.metadata,
        "chunking_strategy": strategy,
        "chunk_index": chunk_index,
        **extract_docling_metadata(chunk),
    }
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return DocumentChunk(
        chunk_id=f"{document.doc_id}_{strategy}_{chunk_index}_{digest}",
        text=text,
        metadata=metadata,
    )


def extract_docling_metadata(chunk: Any) -> Dict[str, Any]:
    """Best-effort metadata extraction from Docling chunk objects."""
    metadata: Dict[str, Any] = {}

    raw_meta = getattr(chunk, "meta", None)
    if raw_meta is None:
        raw_meta = getattr(chunk, "metadata", None)

    headings = getattr(raw_meta, "headings", None)
    if headings:
        metadata["headings"] = list(headings)

    captions = getattr(raw_meta, "captions", None)
    if captions:
        metadata["captions"] = list(captions)

    doc_items = getattr(raw_meta, "doc_items", None)
    pages = []
    item_types = []
    if doc_items:
        for item in doc_items:
            label = getattr(item, "label", None)
            if label is not None:
                item_types.append(str(label))
            for prov in getattr(item, "prov", []) or []:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    pages.append(page_no)

    if pages:
        unique_pages = sorted(set(pages))
        metadata["page_no"] = unique_pages[0] if len(unique_pages) == 1 else ",".join(
            str(page) for page in unique_pages
        )
    else:
        metadata["page_no"] = "unknown"

    if item_types:
        metadata["doc_item_types"] = sorted(set(item_types))

    return metadata
