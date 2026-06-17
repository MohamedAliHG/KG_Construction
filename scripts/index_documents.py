"""
scripts/index_documents.py
--------------------------
Project-level command for indexing raw PDFs into ChromaDB using the
integrated document indexing stack.

This keeps document indexing separate from KG construction while making the
command feel like part of this repository rather than an external add-on.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from ingestion.document_indexing.pipeline import DocumentPipeline, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index PDFs into local ChromaDB using Docling."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Raw PDF file or directory override",
    )
    parser.add_argument(
        "--strategy",
        help="Optional chunking strategy override",
    )
    parser.add_argument(
        "--namespace",
        help="Namespace written into Chroma metadata for later KG loading.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override chunk size for the active chunking strategy",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Override overlap for fixed_character chunking",
    )
    parser.add_argument(
        "--image-resolution-scale",
        type=float,
        help="Docling image resolution scale override",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def _apply_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.input_path is not None:
        settings.document_input_path = args.input_path
    if args.strategy is not None:
        settings.document_chunk_strategy = args.strategy
    if args.namespace is not None:
        settings.chroma_namespace = args.namespace
    effective_strategy = settings.document_chunk_strategy
    if args.chunk_size is not None:
        if effective_strategy == "fixed_character":
            settings.document_fixed_character_chunk_size = args.chunk_size
        elif effective_strategy == "hybrid":
            settings.document_hybrid_max_tokens = args.chunk_size
        else:
            parser.error(
                "--chunk-size only applies to fixed_character or hybrid"
            )
    if args.chunk_overlap is not None:
        if effective_strategy == "fixed_character":
            settings.document_fixed_character_chunk_overlap = args.chunk_overlap
        else:
            parser.error(
                "--chunk-overlap only applies to fixed_character"
            )
    if args.image_resolution_scale is not None:
        settings.document_image_resolution_scale = args.image_resolution_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDFs into local ChromaDB")
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    _apply_overrides(parser, args)

    pipeline = DocumentPipeline.from_settings()
    result = pipeline.run()

    print(
        "Indexed "
        f"{result.chunks_written} chunks from {result.documents_processed} document(s) "
        f"into collection='{result.collection_name}' "
        f"persist_directory='{result.persist_directory}' "
        f"namespace='{result.namespace}'"
    )


if __name__ == "__main__":
    main()
