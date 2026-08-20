from graph.schema_profiles import SchemaLevel, build_schema_profile, load_schema_profile_data


PROFILE_TEXT = """
name: split_schema
extraction:
  allowed_nodes:
    - Source
    - Reference
  allowed_relationships:
    constrained:
      - HAS_REFERENCE
    strict:
      - source: Source
        type: HAS_REFERENCE
        target: Reference
allowed_nodes:
  - Source
  - Reference
  - DerivedTarget
allowed_relationships:
  constrained:
    - HAS_REFERENCE
    - REFERENCES_TARGET
  strict:
    - source: Source
      type: HAS_REFERENCE
      target: Reference
    - source: Reference
      type: REFERENCES_TARGET
      target: DerivedTarget
additional_instructions: "Preserve complete source values."
"""


def test_extraction_schema_is_passed_to_transformer_without_narrowing_final_schema(tmp_path):
    path = tmp_path / "split_schema.yaml"
    path.write_text(PROFILE_TEXT, encoding="utf-8")

    extraction = build_schema_profile(SchemaLevel.STRICT, profile_path=path)
    final_schema = load_schema_profile_data(path)

    assert extraction.allowed_nodes == ("Source", "Reference")
    assert extraction.allowed_relationships == (
        ("Source", "HAS_REFERENCE", "Reference"),
    )
    assert extraction.additional_instructions == "Preserve complete source values."
    assert "DerivedTarget" in final_schema["allowed_nodes"]
    assert any(
        item.get("type") == "REFERENCES_TARGET"
        for item in final_schema["allowed_relationships"]["strict"]
    )
