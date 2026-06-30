from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument

from config import settings

logger = logging.getLogger(__name__)

_graph: Neo4jGraph | None = None

_VECTOR_SIMILARITY_FUNCTIONS = {"cosine", "euclidean", "dot_product"}


def _safe_type_name(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    return cleaned or fallback


def _source_chunk_payload(graph_doc: GraphDocument) -> dict[str, Any] | None:
    if graph_doc.source is None:
        return None

    metadata = dict(graph_doc.source.metadata or {})
    chunk_id = metadata.get("chunk_id") or metadata.get("id")
    if not chunk_id:
        chunk_id = hashlib.md5(graph_doc.source.page_content.encode("utf-8")).hexdigest()
        metadata["chunk_id"] = chunk_id

    embedding = metadata.pop("embedding", None)
    embedding_model = metadata.pop("embedding_model", None)
    embedding_dim = metadata.pop("embedding_dim", None)
    chunk_index = _coerce_int(metadata.get("chunk_index"))
    loaded_index = _coerce_int(metadata.get("loaded_index"))
    sequence_index = chunk_index if chunk_index is not None else loaded_index
    document_id = _resolve_source_document_id(metadata)
    document_name = _resolve_source_document_name(metadata, document_id)

    chunk_metadata = dict(metadata)
    chunk_metadata.pop("id", None)
    chunk_metadata["chunk_id"] = chunk_id
    chunk_metadata["source_document_id"] = document_id

    document_metadata = _source_document_metadata(
        metadata=metadata,
        document_id=document_id,
        document_name=document_name,
    )

    return {
        "document_id": document_id,
        "document_name": document_name,
        "document_metadata": document_metadata,
        "chunk_id": chunk_id,
        "text": graph_doc.source.page_content,
        "chunk_metadata": chunk_metadata,
        "chunk_index": sequence_index,
        "previous_chunk_index": sequence_index - 1 if sequence_index and sequence_index > 0 else None,
        "next_chunk_index": sequence_index + 1 if sequence_index is not None else None,
        "is_first_chunk": sequence_index == 0,
        "embedding": embedding,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }


def _node_rows(graph_doc: GraphDocument) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in graph_doc.nodes:
        rows.append(
            {
                "id": node.id,
                "type": node.type,
                "label": _safe_type_name(node.type, "Entity"),
                "properties": node.properties or {},
            }
        )
    return rows


def _relationship_rows(graph_doc: GraphDocument) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in graph_doc.relationships:
        rows.append(
            {
                "source": rel.source.id,
                "target": rel.target.id,
                "type": rel.type,
                "label": _safe_type_name(rel.type.replace(" ", "_").upper(), "RELATED_TO"),
                "properties": rel.properties or {},
            }
        )
    return rows


def get_graph() -> Neo4jGraph:
    global _graph
    if _graph is None:
        logger.info("Connecting to Neo4j at %s", settings.neo4j_url)
        _graph = Neo4jGraph(
            url=settings.neo4j_url,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            refresh_schema=False,  # avoids requiring APOC
        )
    return _graph


def create_chunk_vector_index(
    index_name: str = "chunk_embedding_index",
    dimensions: int = 384,
    similarity_function: str = "cosine",
) -> None:
    _validate_vector_index_name(index_name)
    _validate_vector_dimensions(dimensions)
    similarity_function = _validate_similarity_function(similarity_function)

    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (c:Chunk)
    ON (c.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dimensions},
        `vector.similarity_function`: '{similarity_function}'
      }}
    }}
    """
    get_graph().query(query)
    logger.info(
        "Created/verified vector index '%s' on (:Chunk).embedding (dimensions=%d, similarity=%s)",
        index_name,
        dimensions,
        similarity_function,
    )


def add_graph_documents(graph_docs: list[GraphDocument]) -> None:
    g = get_graph()

    for graph_doc in graph_docs:
        source_payload = _source_chunk_payload(graph_doc)
        if source_payload is not None:
            g.query(
                """
                MERGE (d:Document {id: $document_id})
                SET d += $document_metadata
                SET d.name = $document_name
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text
                SET c += $chunk_metadata
                FOREACH (_ IN CASE WHEN $chunk_index IS NOT NULL THEN [1] ELSE [] END |
                    SET c.chunk_index = $chunk_index
                )
                FOREACH (_ IN CASE WHEN $embedding IS NOT NULL THEN [1] ELSE [] END |
                    SET c.embedding = $embedding
                )
                FOREACH (_ IN CASE WHEN $embedding_model IS NOT NULL THEN [1] ELSE [] END |
                    SET c.embedding_model = $embedding_model
                )
                FOREACH (_ IN CASE WHEN $embedding_dim IS NOT NULL THEN [1] ELSE [] END |
                    SET c.embedding_dim = $embedding_dim
                )
                MERGE (c)-[:PART_OF]->(d)
                FOREACH (_ IN CASE WHEN $is_first_chunk THEN [1] ELSE [] END |
                    MERGE (d)-[:FIRST_CHUNK]->(c)
                )
                WITH d, c
                OPTIONAL MATCH (d)<-[:PART_OF]-(previous:Chunk)
                WHERE previous.chunk_index = $previous_chunk_index
                FOREACH (_ IN CASE WHEN previous IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (previous)-[:NEXT_CHUNK]->(c)
                )
                WITH d, c
                OPTIONAL MATCH (d)<-[:PART_OF]-(next:Chunk)
                WHERE next.chunk_index = $next_chunk_index
                FOREACH (_ IN CASE WHEN next IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (c)-[:NEXT_CHUNK]->(next)
                )
                """,
                source_payload,
            )

        node_rows = _node_rows(graph_doc)
        if node_rows:
            g.query(
                """
                UNWIND $rows AS row
                MERGE (n:Entity {id: row.id})
                SET n += row.properties
                SET n.type = row.type
                SET n:$(row.label)
                WITH n, row
                FOREACH (_ IN CASE WHEN $has_source THEN [1] ELSE [] END |
                    MERGE (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:HAS_ENTITY]->(n)
                )
                RETURN count(n) AS nodes_written
                """,
                {
                    "rows": node_rows,
                    "has_source": source_payload is not None,
                    "chunk_id": source_payload["chunk_id"] if source_payload else None,
                },
            )

        relationship_rows = _relationship_rows(graph_doc)
        if relationship_rows:
            g.query(
                """
                UNWIND $rows AS row
                MATCH (source:Entity {id: row.source})
                MATCH (target:Entity {id: row.target})
                MERGE (source)-[rel:$(row.label)]->(target)
                SET rel += row.properties
                SET rel.type = row.type
                RETURN count(rel) AS relationships_written
                """,
                {"rows": relationship_rows},
            )

    logger.info("Stored %d graph document(s) to Neo4j", len(graph_docs))


def clean_graph() -> None:
    logger.warning("Deleting all nodes and relationships from Neo4j graph")
    get_graph().query("MATCH (n) DETACH DELETE n")


def refresh_schema() -> str:
    g = get_graph()
    g.refresh_schema()
    return g.schema


def _validate_vector_index_name(index_name: str) -> str:
    if not index_name:
        raise ValueError("index_name must not be empty")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", index_name):
        raise ValueError(
            "index_name must contain only letters, numbers, and underscores, "
            "and must not start with a number"
        )
    return index_name


def _validate_vector_dimensions(dimensions: int) -> int:
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")
    return dimensions


def _validate_similarity_function(similarity_function: str) -> str:
    normalized = (similarity_function or "").strip().lower()
    if normalized not in _VECTOR_SIMILARITY_FUNCTIONS:
        allowed = ", ".join(sorted(_VECTOR_SIMILARITY_FUNCTIONS))
        raise ValueError(
            f"Unsupported similarity_function '{similarity_function}'. Allowed: {allowed}"
        )
    return normalized


def _resolve_source_document_id(metadata: dict[str, Any]) -> str:
    for key in ("source_document_id", "document_id", "doc_id", "source_path", "source", "source_name"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return "document"


def _resolve_source_document_name(metadata: dict[str, Any], document_id: str) -> str:
    for key in ("source_document_name", "source_name", "source", "source_path"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        if key == "source_path":
            return str(value).rstrip("/").split("/")[-1] or document_id
        return str(value)
    return document_id


def _source_document_metadata(
    *,
    metadata: dict[str, Any],
    document_id: str,
    document_name: str,
) -> dict[str, Any]:
    document_metadata = {
        "source_document_id": document_id,
        "source_document_name": document_name,
    }
    for key in (
        "document_id",
        "doc_id",
        "source_path",
        "source_name",
        "source",
        "namespace",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            document_metadata[key] = value
    return document_metadata


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
