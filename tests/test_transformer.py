import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.schema_profiles import (
    DEFAULT_ALLOWED_NODES,
    DEFAULT_ALLOWED_RELATIONSHIPS,
    DEFAULT_STRICT_RELATIONSHIPS,
    ExtractionMode,
    SchemaLevel,
    build_schema_profile,
    parse_property_spec,
)


def test_schema_profiles_cover_all_levels():
    unconstrained = build_schema_profile(SchemaLevel.UNCONSTRAINED)
    constrained = build_schema_profile(SchemaLevel.CONSTRAINED)
    strict = build_schema_profile(SchemaLevel.STRICT)

    assert unconstrained.allowed_nodes == ()
    assert unconstrained.allowed_relationships == ()

    assert constrained.allowed_nodes == DEFAULT_ALLOWED_NODES
    assert constrained.allowed_relationships == DEFAULT_ALLOWED_RELATIONSHIPS

    assert strict.allowed_nodes == DEFAULT_ALLOWED_NODES
    assert strict.allowed_relationships == DEFAULT_STRICT_RELATIONSHIPS


def test_build_transformer_tool_mode_uses_schema_kwargs(monkeypatch):
    import graph.transformer as transformer_module

    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_llm = SimpleNamespace(name="fake-llm")
    monkeypatch.setattr(transformer_module, "LLMGraphTransformer", FakeTransformer)

    transformer_module.build_transformer(
        schema_level=SchemaLevel.STRICT,
        extraction_mode=ExtractionMode.TOOL,
        llm=fake_llm,
    )

    assert captured["llm"] is fake_llm
    assert captured["strict_mode"] is True
    assert captured["allowed_nodes"] == list(DEFAULT_ALLOWED_NODES)
    assert captured["allowed_relationships"] == list(DEFAULT_STRICT_RELATIONSHIPS)
    assert "ignore_tool_usage" not in captured


def test_build_transformer_tool_mode_can_enable_properties(monkeypatch):
    import graph.transformer as transformer_module

    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_llm = SimpleNamespace(name="fake-llm")
    monkeypatch.setattr(transformer_module, "LLMGraphTransformer", FakeTransformer)

    transformer_module.build_transformer(
        schema_level=SchemaLevel.CONSTRAINED,
        extraction_mode=ExtractionMode.TOOL,
        node_properties="any",
        relationship_properties="start_date",
        llm=fake_llm,
    )

    assert captured["node_properties"] is True
    assert captured["relationship_properties"] == ["start_date"]


def test_build_transformer_prompt_mode_sets_ignore_tool_usage(monkeypatch):
    import graph.transformer as transformer_module

    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_llm = SimpleNamespace(name="fake-llm")
    monkeypatch.setattr(transformer_module, "LLMGraphTransformer", FakeTransformer)

    transformer_module.build_transformer(
        schema_level=SchemaLevel.UNCONSTRAINED,
        extraction_mode=ExtractionMode.PROMPT,
        llm=fake_llm,
    )

    assert captured["llm"] is fake_llm
    assert captured["strict_mode"] is True
    assert captured["ignore_tool_usage"] is True
    assert "allowed_nodes" not in captured
    assert "allowed_relationships" not in captured
    assert "node_properties" not in captured
    assert "relationship_properties" not in captured


def test_build_llm_local_provider_uses_openai_client(monkeypatch):
    import graph.transformer as transformer_module

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(transformer_module.settings, "llm_provider", "local")
    monkeypatch.setattr(transformer_module.settings, "local_llm_model", "local-test")
    monkeypatch.setattr(
        transformer_module.settings, "local_llm_base_url", "http://localhost:8080/v1"
    )
    monkeypatch.setattr(transformer_module.settings, "local_llm_api_key", "local-key")

    llm = transformer_module.build_llm()

    assert llm is not None
    assert captured["model"] == "local-test"
    assert captured["base_url"] == "http://localhost:8080/v1"
    assert captured["api_key"] == "local-key"
    assert captured["temperature"] == 0


def test_parse_property_spec_supports_off_any_and_lists():
    assert parse_property_spec(None) is False
    assert parse_property_spec("off") is False
    assert parse_property_spec("any") is True
    assert parse_property_spec(" birth_date, death_date ") == ("birth_date", "death_date")
