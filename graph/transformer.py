from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import replace
from typing import Any

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq

from config import settings
from graph.schema_profiles import (
    ExtractionMode,
    SchemaLevel,
    parse_property_spec,
    build_schema_profile,
    resolve_extraction_mode,
    resolve_schema_level,
)

logger = logging.getLogger(__name__)


def build_llm(provider: str | None = None) -> Any:
    """Construct the configured chat model used for graph extraction."""
    resolved_provider = (provider or settings.llm_provider).lower().strip()
    if resolved_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "groq_api_key is required when llm_provider='groq'"
            )
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
    if resolved_provider == "local":
        return _build_local_llm()
    raise ValueError(f"Unsupported LLM provider: {provider or settings.llm_provider!r}")


def _build_local_llm() -> Any:
    chat_openai_module = importlib.import_module("langchain_openai")
    chat_openai_cls = getattr(chat_openai_module, "ChatOpenAI")
    return chat_openai_cls(
        model=settings.local_llm_model,
        api_key=settings.local_llm_api_key or "local-key",
        base_url=settings.local_llm_base_url,
        temperature=0,
    )


def build_transformer(
    *,
    llm_provider: str | None = None,
    schema_level: str | SchemaLevel | None = None,
    extraction_mode: str | ExtractionMode | None = None,
    node_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    relationship_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    llm: Any | None = None,
) -> LLMGraphTransformer:
    profile = build_schema_profile(resolve_schema_level(schema_level))
    mode = resolve_extraction_mode(extraction_mode)
    if llm is not None:
        model = llm
    else:
        model = build_llm(provider=llm_provider)

    node_prop_spec = parse_property_spec(node_properties)
    rel_prop_spec = parse_property_spec(relationship_properties)
    if mode == ExtractionMode.PROMPT and (node_prop_spec or rel_prop_spec):
        logger.warning(
            "Ignoring node/relationship properties because prompt mode does not support property extraction."
        )
        node_prop_spec = False
        rel_prop_spec = False

    profile = replace(
        profile,
        node_properties=node_prop_spec,
        relationship_properties=rel_prop_spec,
    )

    kwargs = profile.to_transformer_kwargs()
    kwargs["llm"] = model
    if mode == ExtractionMode.PROMPT:
        kwargs["ignore_tool_usage"] = True

    logger.info(
        "Building graph transformer | mode=%s | schema=%s",
        mode.value,
        profile.level.value,
    )
    return LLMGraphTransformer(**kwargs)


async def extract_graph_documents(
    documents: list[Document],
    *,
    llm_provider: str | None = None,
    schema_level: str | SchemaLevel | None = None,
    extraction_mode: str | ExtractionMode | None = None,
    node_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    relationship_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    llm: Any | None = None,
) -> list[GraphDocument]:
    transformer = build_transformer(
        llm_provider=llm_provider,
        schema_level=schema_level,
        extraction_mode=extraction_mode,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
        llm=llm,
    )
    logger.info(
        "Extracting graph from %d document(s) ...", len(documents)
    )
    graph_docs = await transformer.aconvert_to_graph_documents(documents)
    logger.info(
        "Extraction complete -- %d graph document(s) produced", len(graph_docs)
    )
    return graph_docs


def extract_graph_documents_sync(
    documents: list[Document],
    *,
    llm_provider: str | None = None,
    schema_level: str | SchemaLevel | None = None,
    extraction_mode: str | ExtractionMode | None = None,
    node_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    relationship_properties: str | bool | tuple[str, ...] | list[str] | None = None,
    llm: Any | None = None,
) -> list[GraphDocument]:
    return asyncio.run(
        extract_graph_documents(
            documents,
            llm_provider=llm_provider,
            schema_level=schema_level,
            extraction_mode=extraction_mode,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            llm=llm,
        )
    )
