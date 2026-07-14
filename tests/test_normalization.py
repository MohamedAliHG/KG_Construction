import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from graph.normalization import normalize_graph_documents

RELATIONSHIP_PROFILE_TEXT = """
name: relationship_profile
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
normalization:
  preserve_raw: true
  drop_unknown_nodes: true
  drop_unknown_relationships: true
  default_transforms: [fix_pdf_hyphenation, collapse_whitespace, strip]
  nodes:
    Person:
      aliases: [person]
      required_properties: [name]
      allowed_properties: [name]
      id_property: name
      reject_id_patterns: ["^Person\\\\d+$"]
      normalize_properties:
        name: [fix_pdf_hyphenation, collapse_whitespace, strip]
    Organization:
      aliases: [org]
      required_properties: [name]
      allowed_properties: [name]
      id_property: name
      normalize_properties:
        name: [collapse_whitespace, strip]
  relationships:
    WORKS_FOR:
      source: Person
      target: Organization
      allow_reverse: true
      allowed_properties: [role]
      normalize_properties:
        role: [collapse_whitespace, strip]
additional_instructions: ""
"""

DERIVATION_PROFILE_TEXT = """
name: derivation_profile
allowed_nodes:
  - Document
  - Section
allowed_relationships:
  constrained:
    - HAS_SECTION
  strict:
    - source: Document
      type: HAS_SECTION
      target: Section
normalization:
  preserve_raw: true
  drop_unknown_nodes: true
  drop_unknown_relationships: true
  default_transforms: [collapse_whitespace, strip]
  nodes:
    Document:
      required_properties: [document_key, title]
      allowed_properties: [document_key, title]
      id_property: document_key
      derive_properties:
        document_key:
          source: id
          transforms: [colon_case]
        title:
          template: "Document {document_key}"
    Section:
      required_properties: [section_key, document_key, section_number, title]
      allowed_properties: [section_key, document_key, section_number, title]
      id_property: section_key
      derive_properties:
        section_key:
          source: id
          transforms: [colon_case]
        document_key:
          source: section_key
          regex_extract: "^(document:[^:]+):section:[^:]+$"
        section_number:
          source: section_key
          regex_extract: "^document:[^:]+:section:([^:]+)$"
        title:
          template: "Section {section_number}"
  relationships:
    HAS_SECTION:
      source: Document
      target: Section
      allow_reverse: true
additional_instructions: ""
"""


def _write_profile(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_profile_normalization_canonicalizes_nodes_and_relationships(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        "relationship_profile.yaml",
        RELATIONSHIP_PROFILE_TEXT,
    )
    person = Node(
        id="Person1",
        type="person",
        properties={"name": "Alice  Smith", "extra": "ignored"},
    )
    organization = Node(
        id="Org1",
        type="org",
        properties={"name": "Example  Corp"},
    )
    relationship = Relationship(
        source=organization,
        target=person,
        type="works for",
        properties={"role": " Engineer "},
    )
    graph_doc = GraphDocument(
        nodes=[person, organization],
        relationships=[relationship],
        source=Document(page_content="source text"),
    )

    normalized, report = normalize_graph_documents(
        [graph_doc],
        mode="profile",
        schema_profile_path=profile_path,
    )

    result = normalized[0]
    nodes = {(node.type, node.id): node for node in result.nodes}

    assert ("Person", "Alice Smith") in nodes
    assert ("Organization", "Example Corp") in nodes
    assert "extra" not in nodes[("Person", "Alice Smith")].properties

    assert len(result.relationships) == 1
    rel = result.relationships[0]
    assert rel.type == "WORKS_FOR"
    assert rel.source.type == "Person"
    assert rel.target.type == "Organization"
    assert rel.properties["role"] == "Engineer"
    assert report.nodes_renamed > 0
    assert report.relationships_renamed == 1
    assert report.relationships_reversed == 1


def test_profile_normalization_drops_invalid_nodes_and_relationships(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        "relationship_profile.yaml",
        RELATIONSHIP_PROFILE_TEXT,
    )
    bad_person = Node(
        id="Person1",
        type="Person",
        properties={},
    )
    organization = Node(
        id="Org1",
        type="Organization",
        properties={"name": "Example Corp"},
    )
    relationship = Relationship(
        source=bad_person,
        target=organization,
        type="WORKS_FOR",
        properties={},
    )
    graph_doc = GraphDocument(
        nodes=[bad_person, organization],
        relationships=[relationship],
        source=Document(page_content="source text"),
    )

    normalized, report = normalize_graph_documents(
        [graph_doc],
        mode="profile",
        schema_profile_path=profile_path,
    )

    assert len(normalized[0].nodes) == 1
    assert normalized[0].nodes[0].type == "Organization"
    assert normalized[0].nodes[0].id == "Example Corp"
    assert normalized[0].relationships == []
    assert report.nodes_dropped == 1
    assert report.relationships_dropped == 1
    assert report.invalid_endpoint_count == 1


def test_profile_normalization_derives_properties_from_canonical_ids(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        "derivation_profile.yaml",
        DERIVATION_PROFILE_TEXT,
    )
    document = Node(
        id="document:manual-1",
        type="Document",
        properties={},
    )
    section = Node(
        id="document:manual-1:section:2",
        type="Section",
        properties={},
    )
    relationship = Relationship(
        source=document,
        target=section,
        type="HAS_SECTION",
        properties={},
    )
    graph_doc = GraphDocument(
        nodes=[document, section],
        relationships=[relationship],
        source=Document(page_content="source text"),
    )

    normalized, report = normalize_graph_documents(
        [graph_doc],
        mode="profile",
        schema_profile_path=profile_path,
    )

    nodes = {(node.type, node.id): node for node in normalized[0].nodes}

    assert report.nodes_dropped == 0
    assert report.relationships_dropped == 0
    assert nodes[("Document", "document:manual-1")].properties["title"] == "Document document:manual-1"
    assert (
        nodes[("Section", "document:manual-1:section:2")].properties["document_key"]
        == "document:manual-1"
    )
    assert nodes[("Section", "document:manual-1:section:2")].properties["section_number"] == "2"
    assert nodes[("Section", "document:manual-1:section:2")].properties["title"] == "Section 2"

    rels = {
        (rel.type, rel.source.type, rel.target.type): rel
        for rel in normalized[0].relationships
    }
    assert ("HAS_SECTION", "Document", "Section") in rels
