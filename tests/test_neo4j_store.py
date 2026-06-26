import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_create_document_vector_index_validates_name(monkeypatch):
    import graph.neo4j_store as neo4j_store

    class FakeGraph:
        def __init__(self):
            self.queries = []

        def query(self, query, params=None):
            self.queries.append((query, params))

    fake_graph = FakeGraph()
    monkeypatch.setattr(neo4j_store, "get_graph", lambda: fake_graph)

    neo4j_store.create_document_vector_index(
        index_name="document_embedding_index",
        dimensions=384,
        similarity_function="cosine",
    )

    assert fake_graph.queries
    assert "CREATE VECTOR INDEX document_embedding_index IF NOT EXISTS" in fake_graph.queries[0][0]


@pytest.mark.parametrize("name", ["", "1bad", "bad-name", "bad name", "bad.name"])
def test_create_document_vector_index_rejects_bad_names(monkeypatch, name):
    import graph.neo4j_store as neo4j_store

    monkeypatch.setattr(neo4j_store, "get_graph", lambda: None)

    with pytest.raises(ValueError):
        neo4j_store.create_document_vector_index(index_name=name)
