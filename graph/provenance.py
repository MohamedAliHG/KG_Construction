"""Config-driven chunk-to-entity provenance evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class ProvenanceDecision:
    """A chunk-to-entity relationship selected by provenance policy."""

    relationship_type: str
    rule_id: str
    properties: dict[str, Any] = field(default_factory=dict)


def evaluate_provenance(
    *,
    source: Document,
    node: Any,
    config: Mapping[str, Any] | None,
) -> list[ProvenanceDecision]:
    """Return mention plus any evidence-backed description decisions."""
    policy = dict(config or {})
    mention_type = str(policy.get("mention_relationship") or "MENTIONS_ENTITY")
    description_type = str(policy.get("description_relationship") or "DESCRIBES_ENTITY")
    decisions = [ProvenanceDecision(mention_type, "default")]

    for index, raw_rule in enumerate(policy.get("description_rules") or ()):
        if not isinstance(raw_rule, Mapping):
            raise ValueError("provenance.description_rules entries must be mappings")
        rule = dict(raw_rule)
        rule_id = str(rule.get("id") or f"description_rule_{index + 1}")
        entity_types = {str(value) for value in (rule.get("entity_types") or ())}
        if entity_types and str(node.type) not in entity_types:
            continue
        condition = rule.get("when")
        if condition is None:
            raise ValueError(f"Provenance rule '{rule_id}' requires a when condition")
        if not _evaluate_condition(
            condition,
            text=source.page_content or "",
            metadata=source.metadata or {},
            node=node,
        ):
            continue

        properties = dict(rule.get("relationship_properties") or {})
        properties["rule_id"] = rule_id
        decisions.append(ProvenanceDecision(description_type, rule_id, properties))

    return decisions


def _evaluate_condition(
    condition: Any,
    *,
    text: str,
    metadata: Mapping[str, Any],
    node: Any,
) -> bool:
    if not isinstance(condition, Mapping):
        raise ValueError("Provenance conditions must be mappings")

    if "all" in condition:
        children = condition["all"]
        _require_condition_list(children, "all")
        return all(
            _evaluate_condition(child, text=text, metadata=metadata, node=node)
            for child in children
        )
    if "any" in condition:
        children = condition["any"]
        _require_condition_list(children, "any")
        return any(
            _evaluate_condition(child, text=text, metadata=metadata, node=node)
            for child in children
        )
    if "not" in condition:
        return not _evaluate_condition(condition["not"], text=text, metadata=metadata, node=node)
    if "text_regex" in condition:
        pattern = _render_template(str(condition["text_regex"]), node)
        return re.search(pattern, text, flags=re.MULTILINE) is not None
    if "text_contains" in condition:
        needle = _render_template(str(condition["text_contains"]), node, escape=False)
        return needle.casefold() in text.casefold()
    if "metadata_equals" in condition:
        expected = condition["metadata_equals"]
        if not isinstance(expected, Mapping):
            raise ValueError("metadata_equals must be a mapping")
        return all(metadata.get(key) == value for key, value in expected.items())
    if "metadata_in" in condition:
        expected = condition["metadata_in"]
        if not isinstance(expected, Mapping):
            raise ValueError("metadata_in must be a mapping")
        return all(metadata.get(key) in values for key, values in expected.items())
    if "entity_property_exists" in condition:
        return _entity_value(node, str(condition["entity_property_exists"])) not in (None, "")

    raise ValueError(f"Unsupported provenance condition: {sorted(condition)}")


def _require_condition_list(value: Any, operator: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Provenance '{operator}' condition must be a list")


def _render_template(template: str, node: Any, *, escape: bool = True) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(1)
        if not path.startswith("entity."):
            raise ValueError(f"Unsupported provenance template variable: {path}")
        value = _entity_value(node, path.removeprefix("entity."))
        if value in (None, ""):
            # A missing value must make a regex rule fail rather than broaden it.
            return r"(?!)" if escape else ""
        rendered = str(value)
        return re.escape(rendered) if escape else rendered

    return re.sub(r"\{(entity\.[A-Za-z_][A-Za-z0-9_]*)\}", replace, template)


def _entity_value(node: Any, key: str) -> Any:
    if key == "id":
        return node.id
    if key == "type":
        return node.type
    return (node.properties or {}).get(key)
