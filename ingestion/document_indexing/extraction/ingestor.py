"""Docling ingestion and extraction stage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

from ingestion.document_indexing.extraction.converter import build_document_converter
from ingestion.document_indexing.types import ParsedDocument


class DoclingIngestor:
    """Convert raw PDF documents into a clean intermediate representation."""

    def __init__(
        self,
        *,
        input_path: str = "data/raw",
        supported_extensions: Iterable[str] = (".pdf",),
        image_resolution_scale: float = 2.0,
        enable_picture_description: bool = False,
    ) -> None:
        self.input_path = input_path
        self.supported_extensions = tuple(ext.lower() for ext in supported_extensions)
        self.converter = build_document_converter(
            image_resolution_scale=image_resolution_scale,
            enable_picture_description=enable_picture_description,
        )

    def iter_paths(
        self,
        input_path: str | Path | None = None,
        supported_extensions: Iterable[str] | None = None,
    ) -> List[Path]:
        """Return supported files from a file or directory path."""
        path = Path(input_path or self.input_path)
        extensions = {ext.lower() for ext in (supported_extensions or self.supported_extensions)}

        if path.is_file():
            if path.suffix.lower() not in extensions:
                raise ValueError(f"Unsupported file extension: {path.suffix}")
            return [path]

        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        files: List[Path] = []
        for ext in extensions:
            files.extend(path.glob(f"*{ext}"))
        return sorted(file for file in files if file.name != ".DS_Store")

    def parse(self, source_path: str | Path, doc_id: str | None = None) -> ParsedDocument:
        """Convert one document with Docling and return a ParsedDocument."""
        path = Path(source_path)
        result = self.converter.convert(path)
        dl_doc = getattr(result, "document", result)
        markdown = _export_markdown(dl_doc)
        resolved_doc_id = doc_id or _stable_doc_id(path)

        return ParsedDocument(
            doc_id=resolved_doc_id,
            source_path=path,
            dl_doc=dl_doc,
            markdown=markdown,
            metadata={
                "doc_id": resolved_doc_id,
                "source_path": str(path),
                "source_name": path.name,
            },
        )


def _stable_doc_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"doc_{digest}"


def _export_markdown(dl_doc) -> str:
    if hasattr(dl_doc, "export_to_markdown"):
        return dl_doc.export_to_markdown()
    if hasattr(dl_doc, "export_to_text"):
        return dl_doc.export_to_text()
    raise AttributeError("Docling document does not expose export_to_markdown/export_to_text")
