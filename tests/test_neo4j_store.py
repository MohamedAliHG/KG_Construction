import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_create_chunk_vector_index_targets_chunk_embeddings(monkeypatch):
    import graph.neo4j_store as neo4j_store

    class FakeGraph:
        def __init__(self):
            self.queries = []

        def query(self, query, params=None):
            self.queries.append((query, params))

    fake_graph = FakeGraph()
    monkeypatch.setattr(neo4j_store, "get_graph", lambda: fake_graph)

    neo4j_store.create_chunk_vector_index(
        index_name="chunk_embedding_index",
        dimensions=384,
        similarity_function="cosine",
    )

    assert fake_graph.queries
    query = fake_graph.queries[0][0]
    assert "CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS" in query
    assert "FOR (c:Chunk)" in query
    assert "ON (c.embedding)" in query


@pytest.mark.parametrize("name", ["", "1bad", "bad-name", "bad name", "bad.name"])
def test_create_chunk_vector_index_rejects_bad_names(monkeypatch, name):
    import graph.neo4j_store as neo4j_store

    monkeypatch.setattr(neo4j_store, "get_graph", lambda: None)

    with pytest.raises(ValueError):
        neo4j_store.create_chunk_vector_index(index_name=name)


def test_source_chunk_payload_separates_document_chunk_and_embedding():
    from langchain_core.documents import Document
    from langchain_community.graphs.graph_document import GraphDocument

    import graph.neo4j_store as neo4j_store

    graph_doc = GraphDocument(
        nodes=[],
        relationships=[],
        source=Document(
            page_content="Chunk text",
            metadata={
                "id": "chunk-1",
                "source": "manual.pdf",
                "chunk_index": 2,
                "embedding": [0.1, 0.2],
                "embedding_model": "test-model",
                "embedding_dim": 2,
            },
        ),
    )

    payload = neo4j_store._source_chunk_payload(graph_doc)

    assert payload["document_id"] == "manual.pdf"
    assert payload["document_name"] == "manual.pdf"
    assert payload["chunk_id"] == "chunk-1"
    assert payload["chunk_index"] == 2
    assert payload["previous_chunk_index"] == 1
    assert payload["next_chunk_index"] == 3
    assert payload["embedding"] == [0.1, 0.2]
    assert payload["embedding_model"] == "test-model"
    assert payload["embedding_dim"] == 2
    assert "embedding" not in payload["chunk_metadata"]


def test_add_graph_documents_links_chunks_to_entities(monkeypatch):
    from langchain_core.documents import Document
    from langchain_community.graphs.graph_document import GraphDocument, Node

    import graph.neo4j_store as neo4j_store

    class FakeGraph:
        def __init__(self):
            self.queries = []

        def query(self, query, params=None):
            self.queries.append((query, params))

    fake_graph = FakeGraph()
    monkeypatch.setattr(neo4j_store, "get_graph", lambda: fake_graph)

    graph_doc = GraphDocument(
        nodes=[Node(id="Entity 1", type="Entity", properties={"name": "Entity 1"})],
        relationships=[],
        source=Document(
            page_content="Chunk text",
            metadata={
                "id": "chunk-1",
                "source_document_id": "doc-1",
                "source_document_name": "manual.pdf",
                "chunk_index": 0,
            },
        ),
    )

    neo4j_store.add_graph_documents([graph_doc])

    queries = "\n".join(query for query, _params in fake_graph.queries)
    assert "MERGE (c)-[:PART_OF]->(d)" in queries
    assert "MERGE (d)-[:FIRST_CHUNK]->(c)" in queries
    assert "MERGE (c)-[:HAS_ENTITY]->(n)" in queries
    assert "MENTIONS" not in queries
