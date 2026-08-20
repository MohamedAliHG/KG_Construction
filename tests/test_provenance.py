import pytest
from langchain_community.graphs.graph_document import Node
from langchain_core.documents import Document

from graph.provenance import evaluate_provenance


POLICY = {
    "mention_relationship": "MENTIONS_ENTITY",
    "description_relationship": "DESCRIBES_ENTITY",
    "description_rules": [
        {
            "id": "preparation_heading",
            "entity_types": ["Preparation"],
            "when": {
                "all": [
                    {"entity_property_exists": "figure_number"},
                    {"entity_property_exists": "letter"},
                    {
                        "text_regex": (
                            r"(?im)^\s*figure\s+{entity.figure_number}\b"
                            r"[^\n]{0,100}\bpreparation\s+{entity.letter}\b"
                        )
                    },
                ]
            },
            "relationship_properties": {"confidence": 1.0},
        }
    ],
}


def preparation_node():
    return Node(
        id="figure:2-1:preparation:a",
        type="Preparation",
        properties={"figure_number": "2-1", "letter": "A"},
    )


def test_provenance_always_emits_mention():
    decisions = evaluate_provenance(
        source=Document(page_content="Unrelated text"),
        node=preparation_node(),
        config=POLICY,
    )

    assert [decision.relationship_type for decision in decisions] == ["MENTIONS_ENTITY"]


def test_provenance_marks_explicit_heading_as_description():
    decisions = evaluate_provenance(
        source=Document(page_content="Figure 2-1. Preparation A\nOpen the access panel."),
        node=preparation_node(),
        config=POLICY,
    )

    assert [decision.relationship_type for decision in decisions] == [
        "MENTIONS_ENTITY",
        "DESCRIBES_ENTITY",
    ]
    assert decisions[1].rule_id == "preparation_heading"
    assert decisions[1].properties["confidence"] == 1.0


def test_provenance_keeps_table_reference_as_mention_only():
    decisions = evaluate_provenance(
        source=Document(
            page_content=(
                "| Fault isolation reference | Perform figure 2-1, Preparation A, "
                "then go to figure 2-9, block 1 |"
            )
        ),
        node=preparation_node(),
        config=POLICY,
    )

    assert [decision.relationship_type for decision in decisions] == ["MENTIONS_ENTITY"]


def test_missing_template_property_fails_closed():
    node = Node(id="preparation:a", type="Preparation", properties={"letter": "A"})
    decisions = evaluate_provenance(
        source=Document(page_content="Figure 2-1. Preparation A"),
        node=node,
        config=POLICY,
    )

    assert [decision.relationship_type for decision in decisions] == ["MENTIONS_ENTITY"]


def test_invalid_condition_is_rejected():
    with pytest.raises(ValueError, match="Unsupported provenance condition"):
        evaluate_provenance(
            source=Document(page_content="text"),
            node=preparation_node(),
            config={
                "description_rules": [
                    {"id": "bad", "when": {"unknown_operator": "value"}}
                ]
            },
        )
