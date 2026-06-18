"""High-level document indexing orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import settings
from ingestion.document_indexing.chunking.registry import create_chunker
from ingestion.document_indexing.embedding import HuggingFaceEmbedder
from ingestion.document_indexing.extraction.ingestor import DoclingIngestor
from ingestion.document_indexing.types import PipelineResult
from ingestion.document_indexing.storage import ChromaWriter

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Raw PDF to ChromaDB pipeline with swappable chunking strategy."""

    def __init__(self) -> None:
        self.ingestor = DoclingIngestor(
            input_path=settings.document_input_path,
            supported_extensions=(".pdf",),
            image_resolution_scale=settings.document_image_resolution_scale,
            enable_picture_description=settings.document_enable_picture_description,
            vlm_url=settings.document_vlm_url,
            vlm_model_name=settings.document_vlm_model_name,
            vlm_timeout=settings.document_vlm_timeout,
            vlm_prompt=settings.document_vlm_prompt,
        )
        self.chunker = create_chunker(settings.document_chunk_strategy)
        self.embedder = HuggingFaceEmbedder()
        self.writer = ChromaWriter()

    @classmethod
    def from_settings(cls) -> "DocumentPipeline":
        return cls()

    def run(
        self,
        input_path: Optional[str | Path] = None,
        strategy: Optional[str] = None,
    ) -> PipelineResult:
        """Run ingestion, chunking, embedding, and ChromaDB writing."""
        if strategy:
            self.chunker = create_chunker(strategy)

        files = self.ingestor.iter_paths(input_path)

        total_chunks = 0
        for file_path in files:
            logger.info("Parsing %s", file_path)
            parsed = self.ingestor.parse(file_path)
            chunks = self.chunker.chunk(parsed)
            logger.info(
                "Chunked %s -> %s chunks using strategy=%s",
                file_path.name,
                len(chunks),
                self.chunker.name,
            )
            embedded = self.embedder.embed_chunks(chunks)
            total_chunks += self.writer.write(embedded)

        return PipelineResult(
            documents_processed=len(files),
            chunks_written=total_chunks,
            collection_name=self.writer.collection_name,
            persist_directory=self.writer.persist_directory,
            namespace=self.writer.namespace,
        )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
