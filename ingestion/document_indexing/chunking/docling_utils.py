"""Helpers for Docling-native chunkers."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, Dict

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import ImageRefMode
from transformers import AutoTokenizer

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


def build_huggingface_tokenizer(model_id: str, max_tokens: int) -> HuggingFaceTokenizer:
    """Load and wrap a Hugging Face tokenizer for Docling chunking."""

    return HuggingFaceTokenizer(tokenizer=_load_auto_tokenizer(model_id), max_tokens=max_tokens)


def build_markdown_serializer_provider(
    *,
    image_mode: str | ImageRefMode,
    image_placeholder: str,
    mark_annotations: bool,
    include_annotations: bool,
) -> ChunkingSerializerProvider:
    """Create a Docling markdown serializer provider with project-specific defaults."""

    resolved_image_mode = _coerce_image_mode(image_mode)

    class MarkdownSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):  # type: ignore[override]
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(
                    image_mode=resolved_image_mode,
                    image_placeholder=image_placeholder,
                    mark_annotations=mark_annotations,
                    include_annotations=include_annotations,
                ),
            )

    return MarkdownSerializerProvider()


@lru_cache(maxsize=8)
def _load_auto_tokenizer(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - depends on local cache/network
        raise RuntimeError(
            f"Unable to load Hugging Face tokenizer '{model_id}'. "
            "Install the model locally or ensure the tokenizer cache is available."
        ) from exc


def _coerce_image_mode(image_mode: str | ImageRefMode) -> ImageRefMode:
    if isinstance(image_mode, ImageRefMode):
        return image_mode
    try:
        return ImageRefMode(str(image_mode))
    except Exception as exc:  # pragma: no cover - invalid config guard
        raise ValueError(f"Invalid Docling image mode: {image_mode!r}") from exc
