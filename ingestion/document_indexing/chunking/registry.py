"""Chunker registry and factory."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Type

from config import settings
from ingestion.document_indexing.chunking.base import BaseChunker

CHUNKER_REGISTRY: Dict[str, Type[BaseChunker]] = {}


def register_chunker(name: str):
    """Register a chunker class under ``name``."""

    def decorator(cls: Type[BaseChunker]) -> Type[BaseChunker]:
        CHUNKER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def create_chunker(strategy: str | None = None) -> BaseChunker:
    """Instantiate a chunker selected by config."""
    _ensure_builtin_chunkers_registered()

    strategy = strategy or settings.document_chunk_strategy
    chunker_cls = CHUNKER_REGISTRY.get(strategy)
    if chunker_cls is None:
        available = ", ".join(sorted(CHUNKER_REGISTRY))
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Available: {available}")

    if strategy == "fixed_character":
        return chunker_cls(
            chunk_size=settings.document_fixed_character_chunk_size,
            chunk_overlap=settings.document_fixed_character_chunk_overlap,
        )
    if strategy == "hierarchical":
        return chunker_cls(
            merge_list_items=settings.document_hierarchical_merge_list_items,
        )
    if strategy == "hybrid":
        return chunker_cls(
            tokenizer=settings.document_hybrid_tokenizer,
            max_tokens=settings.document_hybrid_max_tokens,
            merge_peers=settings.document_hybrid_merge_peers,
            image_mode=settings.document_hybrid_image_mode,
            image_placeholder=settings.document_hybrid_image_placeholder,
            mark_annotations=settings.document_hybrid_mark_annotations,
            include_annotations=settings.document_hybrid_include_annotations,
        )

    raise ValueError(f"Missing config block for chunking strategy '{strategy}'")


def _ensure_builtin_chunkers_registered() -> None:
    """Auto-import chunking modules so decorators can register strategies."""
    import ingestion.document_indexing.chunking as chunking_pkg

    skip_modules = {"base", "registry"}
    for module_info in pkgutil.iter_modules(chunking_pkg.__path__):
        if module_info.name in skip_modules or module_info.name.startswith("_"):
            continue
        importlib.import_module(f"{chunking_pkg.__name__}.{module_info.name}")
