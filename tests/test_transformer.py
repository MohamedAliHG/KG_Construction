import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.schema_profiles import (
    ExtractionMode,
    SchemaLevel,
    build_schema_profile,
    load_schema_profile_data,
    parse_property_spec,
)

GENERIC_PROFILE = "config/schema_profiles/generic.yaml"
STRICT_PROFILE_TEXT = """
name: test_profile
allowed_nodes:
  - Person
  - Organization
allowed_relationships:
  constrained:
    - WORKS_FOR
  strict:
    - source: Person
      type: WORKS_FOR
      target: Organization
additional_instructions: "Extract employment relationships."
"""
EXPECTED_ALLOWED_NODES = (
    "Person",
    "Organization",
)
EXPECTED_ALLOWED_RELATIONSHIPS = (
    "WORKS_FOR",
)
EXPECTED_STRICT_RELATIONSHIPS = (
    ("Person", "WORKS_FOR", "Organization"),
)


@pytest.fixture()
def strict_profile_path(tmp_path):
    path = tmp_path / "strict_profile.yaml"
    path.write_text(STRICT_PROFILE_TEXT, encoding="utf-8")
    return str(path)


def test_schema_profiles_cover_all_levels(strict_profile_path):
    unconstrained = build_schema_profile(SchemaLevel.UNCONSTRAINED)
    constrained = build_schema_profile(SchemaLevel.CONSTRAINED, profile_path=strict_profile_path)
    strict = build_schema_profile(SchemaLevel.STRICT, profile_path=strict_profile_path)

    assert unconstrained.allowed_nodes == ()
    assert unconstrained.allowed_relationships == ()

    assert constrained.allowed_nodes == EXPECTED_ALLOWED_NODES
    assert constrained.allowed_relationships == EXPECTED_ALLOWED_RELATIONSHIPS

    assert strict.allowed_nodes == EXPECTED_ALLOWED_NODES
    assert strict.allowed_relationships == EXPECTED_STRICT_RELATIONSHIPS
    assert "employment relationships" in strict.additional_instructions


def test_load_schema_profile_data_reads_yaml():
    data = load_schema_profile_data(GENERIC_PROFILE)

    assert data["name"] == "generic"
    assert data["allowed_nodes"] == []
    assert "normalization" in data


def test_build_transformer_tool_mode_uses_schema_kwargs(monkeypatch, strict_profile_path):
    import graph.transformer as transformer_module

    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_llm = SimpleNamespace(name="fake-llm")
    monkeypatch.setattr(transformer_module, "LLMGraphTransformer", FakeTransformer)

    transformer_module.build_transformer(
        schema_level=SchemaLevel.STRICT,
        schema_profile_path=strict_profile_path,
        extraction_mode=ExtractionMode.TOOL,
        llm=fake_llm,
    )

    assert captured["llm"] is fake_llm
    assert captured["strict_mode"] is True
    assert captured["allowed_nodes"] == list(EXPECTED_ALLOWED_NODES)
    assert captured["allowed_relationships"] == list(EXPECTED_STRICT_RELATIONSHIPS)
    assert "ignore_tool_usage" not in captured


def test_build_transformer_tool_mode_can_enable_properties(monkeypatch, strict_profile_path):
    import graph.transformer as transformer_module

    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_llm = SimpleNamespace(name="fake-llm")
    monkeypatch.setattr(transformer_module, "LLMGraphTransformer", FakeTransformer)

    transformer_module.build_transformer(
        schema_level=SchemaLevel.CONSTRAINED,
        schema_profile_path=strict_profile_path,
        extraction_mode=ExtractionMode.TOOL,
        node_properties="any",
        relationship_properties="start_date",
        llm=fake_llm,
    )

    assert captured["node_properties"] is True
    assert captured["relationship_properties"] == ["start_date"]
    assert "properties\" to be a list of key/value objects" in captured["additional_instructions"]


def test_build_transformer_does_not_add_property_format_when_properties_off(monkeypatch, strict_profile_path):
    import graph.transformer as transformer_module

    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_llm = SimpleNamespace(name="fake-llm")
    monkeypatch.setattr(transformer_module, "LLMGraphTransformer", FakeTransformer)

    transformer_module.build_transformer(
        schema_level=SchemaLevel.STRICT,
        schema_profile_path=strict_profile_path,
        extraction_mode=ExtractionMode.TOOL,
        node_properties="off",
        relationship_properties="off",
        llm=fake_llm,
    )

    assert "Tool-call property format requirement" not in captured["additional_instructions"]


def test_property_format_instructions_are_prompt_template_safe():
    from langchain_core.prompts import ChatPromptTemplate

    import graph.transformer as transformer_module

    instructions = transformer_module._with_property_tool_format_instructions(
        "Domain instructions.",
        enabled=True,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("human", instructions + " Use the following input: {input}")]
    )

    rendered = prompt.format(input="sample")

    assert "Tool-call property format requirement" in rendered
    assert "sample" in rendered


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
