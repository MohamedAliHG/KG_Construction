from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from config import settings
from ingestion import load_chunks
from graph import add_graph_documents, extract_graph_documents
from graph.schema_profiles import ExtractionMode, SchemaLevel

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    batches_processed: int = 0
    chunks_processed: int = 0
    graph_docs_created: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        logger.info(
            "Pipeline complete | batches=%d | chunks=%d | graph_docs=%d | time=%.1fs | errors=%d",
            self.batches_processed,
            self.chunks_processed,
            self.graph_docs_created,
            self.elapsed_seconds,
            len(self.errors),
        )
        if self.errors:
            for err in self.errors:
                logger.error("  ✗ %s", err)


async def run_async(
    collection_name: str | None = None,
    batch_size: int | None = None,
    namespace: str | None = None,
    pages: tuple[int, ...] | list[int] | None = None,
    llm_provider: str | None = None,
    schema_level: str | SchemaLevel | None = None,
    extraction_mode: str | ExtractionMode | None = None,
    node_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    relationship_properties: str | bool | tuple[str, ...] | list[str] | None = None,
) -> PipelineStats:
 
    stats = PipelineStats()
    start = time.perf_counter()
    resolved_schema_level = settings.schema_level if schema_level is None else schema_level
    resolved_extraction_mode = (
        settings.extraction_mode if extraction_mode is None else extraction_mode
    )
    resolved_namespace = settings.chroma_namespace if namespace is None else namespace
    resolved_node_properties = (
        settings.node_properties if node_properties is None else node_properties
    )
    resolved_relationship_properties = (
        settings.relationship_properties
        if relationship_properties is None
        else relationship_properties
    )

    for batch in load_chunks(
        collection_name=collection_name,
        batch_size=batch_size,
        namespace=resolved_namespace,
        pages=pages,
    ):
        batch_num = stats.batches_processed + 1
        logger.info("Processing batch %d (%d chunks) …", batch_num, len(batch))

        try:
            graph_docs = await extract_graph_documents(
                batch,
                llm_provider=llm_provider,
                schema_level=resolved_schema_level,
                extraction_mode=resolved_extraction_mode,
                node_properties=resolved_node_properties,
                relationship_properties=resolved_relationship_properties,
            )
            add_graph_documents(graph_docs)

            stats.batches_processed += 1
            stats.chunks_processed += len(batch)
            stats.graph_docs_created += len(graph_docs)

        except Exception as exc: 
            msg = f"Batch {batch_num} failed: {exc}"
            logger.exception(msg)
            stats.errors.append(msg)

    stats.elapsed_seconds = time.perf_counter() - start
    stats.log_summary()
    return stats


def run(
    collection_name: str | None = None,
    batch_size: int | None = None,
    namespace: str | None = None,
    pages: tuple[int, ...] | list[int] | None = None,
    llm_provider: str | None = None,
    schema_level: str | SchemaLevel | None = None,
    extraction_mode: str | ExtractionMode | None = None,
    node_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    relationship_properties: str | bool | tuple[str, ...] | list[str] | None = None,
) -> PipelineStats:
    return asyncio.run(
        run_async(
            collection_name=collection_name,
            batch_size=batch_size,
            namespace=namespace,
            pages=pages,
            llm_provider=llm_provider,
            schema_level=schema_level,
            extraction_mode=extraction_mode,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
        )
    )
