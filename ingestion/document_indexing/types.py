"""Shared data contracts for the document indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParsedDocument:
    """Structure-preserving output of the ingestion stage."""

    doc_id: str
    source_path: Path
    dl_doc: Any
    markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Chunk object passed from chunking to embedding/storage."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """Chunk plus embedding vector ready for vector-store insertion."""

    chunk: DocumentChunk
    embedding: List[float]


@dataclass
class PipelineResult:
    """Summary returned by a pipeline run."""

    documents_processed: int
    chunks_written: int
    collection_name: str
    persist_directory: str
    namespace: Optional[str] = None
