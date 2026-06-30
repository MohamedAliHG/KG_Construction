import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture()
def populated_chroma(tmp_path, monkeypatch):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.create_collection("test_collection")
    emb1 = [0.1] * 384
    emb2 = [0.2] * 384
    col.add(
        documents=["Marie Curie won the Nobel Prize.", "Einstein developed relativity."],
        metadatas=[
            {"source": "doc1", "namespace": "experiment_a", "page_no": 5},
            {"source": "doc2", "namespace": "experiment_b", "page_no": 17},
        ],
        ids=["1", "2"],
        embeddings=[emb1, emb2],
    )

    # Patch settings to point at our temp DB
    from config import settings
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path))
    monkeypatch.setattr(settings, "chroma_collection", "test_collection")
    monkeypatch.setattr(settings, "batch_size", 5)
    monkeypatch.setattr(settings, "chroma_namespace", None)

    return col


def test_load_chunks_yields_documents(populated_chroma):
    from ingestion.loader import load_chunks
    from langchain_core.documents import Document

    batches = list(load_chunks())
    assert len(batches) == 1  

    docs = batches[0]
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)


def test_load_chunks_preserves_metadata(populated_chroma):
    from ingestion.loader import load_chunks

    docs = list(load_chunks())[0]
    sources = {d.metadata.get("source") for d in docs}
    assert sources == {"doc1", "doc2"}
    for doc in docs:
        assert doc.metadata["id"] in {"1", "2"}
        assert doc.metadata["chunk_id"] in {"1", "2"}
        assert doc.metadata["source_document_id"] in {"doc1", "doc2"}
        assert doc.metadata["source_document_name"] in {"doc1", "doc2"}
        assert isinstance(doc.metadata["chunk_index"], int)
        assert isinstance(doc.metadata["loaded_index"], int)
        assert doc.metadata["embedding_model"]
        assert doc.metadata["embedding_dim"] == 384
        assert len(doc.metadata["embedding"]) == 384


def test_load_all_chunks(populated_chroma):
    from ingestion.loader import load_all_chunks

    docs = load_all_chunks()
    assert len(docs) == 2


def test_load_chunks_batching(populated_chroma, monkeypatch):
    from config import settings
    from ingestion.loader import load_chunks

    monkeypatch.setattr(settings, "batch_size", 1)
    batches = list(load_chunks(batch_size=1))
    assert len(batches) == 2


def test_load_chunks_filters_namespace(populated_chroma):
    from ingestion.loader import load_chunks

    batches = list(load_chunks(namespace="experiment_a"))
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0].metadata["namespace"] == "experiment_a"


def test_load_chunks_filters_pages(populated_chroma):
    from ingestion.loader import load_chunks

    batches = list(load_chunks(pages=(17,)))
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0].metadata["page_no"] == 17


def test_load_chunks_filters_namespace_and_pages(populated_chroma):
    from ingestion.loader import load_chunks

    batches = list(load_chunks(namespace="experiment_a", pages=(5, 17)))
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0].metadata["namespace"] == "experiment_a"
    assert batches[0][0].metadata["page_no"] == 5


def test_load_chunks_handles_missing_embeddings(tmp_path, monkeypatch):
    from ingestion.loader import load_chunks

    class FakeCollection:
        def count(self):
            return 1

        def get(self, limit=None, offset=None, include=None, where=None):
            return {
                "ids": ["1"],
                "documents": ["No embedding here."],
                "metadatas": [{"source": "doc1", "namespace": "experiment_a"}],
                "embeddings": [None],
            }

    class FakeClient:
        def get_collection(self, name):
            return FakeCollection()

    monkeypatch.setattr("ingestion.loader._chroma_client", lambda: FakeClient())

    docs = list(load_chunks(collection_name="test_collection_missing_embeddings"))[0]
    assert len(docs) == 1
    assert "embedding" not in docs[0].metadata
    assert "embedding_model" not in docs[0].metadata
    assert "embedding_dim" not in docs[0].metadata
