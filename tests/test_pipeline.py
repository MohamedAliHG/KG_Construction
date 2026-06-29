import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_run_async_threads_schema_and_mode(monkeypatch):
    import pipeline.build_graph as pipeline_module

    async def fake_extract_graph_documents(
        documents,
        *,
        llm_provider,
        schema_level,
        schema_profile_path,
        extraction_mode,
        node_properties,
        relationship_properties,
        llm=None,
    ):
        assert schema_level == "strict"
        assert schema_profile_path == "config/schema_profiles/aviation_maintenance.yaml"
        assert extraction_mode == "prompt"
        assert llm_provider == "local"
        assert node_properties == "off"
        assert relationship_properties == "off"
        assert llm is None
        return [object(), object()]

    def fake_load_chunks(collection_name=None, batch_size=None, namespace=None, pages=None):
        assert collection_name == "demo"
        assert batch_size == 2
        assert namespace == "experiment_a"
        assert pages == (5, 17)
        yield [1, 2]

    captured = []

    def fake_add_graph_documents(graph_docs):
        captured.append(list(graph_docs))

    monkeypatch.setattr(pipeline_module, "load_chunks", fake_load_chunks)
    monkeypatch.setattr(pipeline_module, "extract_graph_documents", fake_extract_graph_documents)
    monkeypatch.setattr(pipeline_module, "add_graph_documents", fake_add_graph_documents)

    stats = asyncio.run(
        pipeline_module.run_async(
            collection_name="demo",
            batch_size=2,
            namespace="experiment_a",
            pages=(5, 17),
            llm_provider="local",
            schema_level="strict",
            schema_profile_path="config/schema_profiles/aviation_maintenance.yaml",
            extraction_mode="prompt",
            node_properties="off",
            relationship_properties="off",
        )
    )

    assert stats.batches_processed == 1
    assert stats.chunks_processed == 2
    assert stats.graph_docs_created == 2
    assert stats.errors == []
    assert len(captured) == 1
    assert len(captured[0]) == 2
