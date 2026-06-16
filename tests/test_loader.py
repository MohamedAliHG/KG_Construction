import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture()
def populated_chroma(tmp_path, monkeypatch):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.create_collection("test_collection")
    col.add(
        documents=["Marie Curie won the Nobel Prize.", "Einstein developed relativity."],
        metadatas=[{"source": "doc1"}, {"source": "doc2"}],
        ids=["1", "2"],
    )

    # Patch settings to point at our temp DB
    from config import settings
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path))
    monkeypatch.setattr(settings, "chroma_collection", "test_collection")
    monkeypatch.setattr(settings, "batch_size", 5)

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
