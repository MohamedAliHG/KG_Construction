from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from string import Formatter
from typing import Any

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from config import settings
from graph.schema_profiles import load_schema_profile_data

logger = logging.getLogger(__name__)


class NormalizationMode(StrEnum):
    OFF = "off"
    BASIC = "basic"
    PROFILE = "profile"


@dataclass
class NormalizationReport:
    graph_docs_seen: int = 0
    nodes_seen: int = 0
    nodes_kept: int = 0
    nodes_dropped: int = 0
    nodes_renamed: int = 0
    nodes_merged: int = 0
    relationships_seen: int = 0
    relationships_kept: int = 0
    relationships_dropped: int = 0
    relationships_renamed: int = 0
    relationships_reversed: int = 0
    missing_required_property_count: int = 0
    invalid_endpoint_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        if len(self.warnings) < 50:
            self.warnings.append(message)

    def log_summary(self) -> None:
        logger.info(
            "Normalization complete | graph_docs=%d | nodes=%d kept/%d seen, dropped=%d, merged=%d | "
            "relationships=%d kept/%d seen, dropped=%d, reversed=%d",
            self.graph_docs_seen,
            self.nodes_kept,
            self.nodes_seen,
            self.nodes_dropped,
            self.nodes_merged,
            self.relationships_kept,
            self.relationships_seen,
            self.relationships_dropped,
            self.relationships_reversed,
        )
        for warning in self.warnings:
            logger.warning("Normalization: %s", warning)


def normalize_graph_documents(
    graph_docs: list[GraphDocument],
    *,
    mode: str | NormalizationMode | None = None,
    schema_profile_path: str | None = None,
) -> tuple[list[GraphDocument], NormalizationReport]:
    resolved_mode = _resolve_mode(mode)
    report = NormalizationReport(graph_docs_seen=len(graph_docs))

    if resolved_mode == NormalizationMode.OFF:
        report.nodes_kept = sum(len(doc.nodes) for doc in graph_docs)
        report.nodes_seen = report.nodes_kept
        report.relationships_kept = sum(len(doc.relationships) for doc in graph_docs)
        report.relationships_seen = report.relationships_kept
        return graph_docs, report

    config = _load_normalization_config(resolved_mode, schema_profile_path)
    normalized = [
        _normalize_graph_document(graph_doc, config=config, report=report)
        for graph_doc in graph_docs
    ]
    report.log_summary()
    return normalized, report


def _resolve_mode(value: str | NormalizationMode | None) -> NormalizationMode:
    if value is None:
        value = getattr(settings, "normalization_mode", "off")
    if isinstance(value, NormalizationMode):
        return value
    return NormalizationMode(str(value).strip().lower())


def _load_normalization_config(
    mode: NormalizationMode,
    schema_profile_path: str | None,
) -> dict[str, Any]:
    if mode == NormalizationMode.BASIC:
        return {
            "preserve_raw": True,
            "drop_unknown_nodes": False,
            "drop_unknown_relationships": False,
            "default_transforms": ["fix_pdf_hyphenation", "collapse_whitespace", "strip"],
        }

    profile_path = schema_profile_path or settings.schema_profile_path
    profile_data = load_schema_profile_data(profile_path)
    config = profile_data.get("normalization") or {}
    if not isinstance(config, dict):
        raise ValueError("normalization must be a mapping in the schema profile")

    return {
        "preserve_raw": True,
        "drop_unknown_nodes": False,
        "drop_unknown_relationships": False,
        "default_transforms": ["fix_pdf_hyphenation", "collapse_whitespace", "strip"],
        **config,
    }


def _normalize_graph_document(
    graph_doc: GraphDocument,
    *,
    config: dict[str, Any],
    report: NormalizationReport,
) -> GraphDocument:
    raw_to_normalized: dict[tuple[str, str], Node] = {}
    canonical_nodes: dict[tuple[str, str], Node] = {}
    normalized_nodes: list[Node] = []

    for node in graph_doc.nodes:
        report.nodes_seen += 1
        normalized = _normalize_node(node, config=config, report=report)
        raw_key = _node_key(node)
        if normalized is None:
            report.nodes_dropped += 1
            continue

        raw_to_normalized[raw_key] = normalized
        canonical_key = _node_key(normalized)
        existing = canonical_nodes.get(canonical_key)
        if existing is not None:
            existing.properties.update(_merge_properties(existing.properties, normalized.properties))
            raw_to_normalized[raw_key] = existing
            report.nodes_merged += 1
            continue

        canonical_nodes[canonical_key] = normalized
        normalized_nodes.append(normalized)
        report.nodes_kept += 1

    normalized_relationships = _normalize_relationships(
        graph_doc.relationships,
        config=config,
        raw_to_normalized=raw_to_normalized,
        report=report,
    )

    return GraphDocument(
        nodes=normalized_nodes,
        relationships=normalized_relationships,
        source=graph_doc.source,
    )


def _normalize_node(
    node: Node,
    *,
    config: dict[str, Any],
    report: NormalizationReport,
) -> Node | None:
    raw_id = str(node.id)
    raw_type = str(node.type)
    type_name, rule = _resolve_node_type(raw_type, config)
    if rule is None and config.get("drop_unknown_nodes"):
        report.add_warning(f"Dropped node with unknown type: {raw_type}")
        return None

    properties = _normalize_properties(
        node.properties or {},
        rule=rule,
        config=config,
        section="nodes",
    )
    properties = _derive_properties(
        properties,
        rule=rule,
        context={"id": raw_id, "type": type_name, **properties},
    )
    node_id = _build_node_id(raw_id, properties, rule, config)

    if _matches_any(node_id, _as_list(rule.get("reject_id_patterns") if rule else None)):
        report.add_warning(f"Dropped {type_name} node with rejected id: {node_id}")
        return None

    missing = _missing_required(properties, rule.get("required_properties") if rule else None)
    if missing:
        report.missing_required_property_count += 1
        report.add_warning(
            f"Dropped {type_name} node '{node_id}' missing required properties: {', '.join(missing)}"
        )
        return None

    if config.get("preserve_raw", True):
        actions = []
        if node_id != raw_id:
            properties.setdefault("raw_id", raw_id)
            actions.append(f"id:{raw_id}->{node_id}")
            report.nodes_renamed += 1
        if type_name != raw_type:
            properties.setdefault("raw_type", raw_type)
            actions.append(f"type:{raw_type}->{type_name}")
            report.nodes_renamed += 1
        if actions:
            properties["normalization_actions"] = actions

    return Node(id=node_id, type=type_name, properties=properties)


def _normalize_relationships(
    relationships: list[Relationship],
    *,
    config: dict[str, Any],
    raw_to_normalized: dict[tuple[str, str], Node],
    report: NormalizationReport,
) -> list[Relationship]:
    normalized_relationships: list[Relationship] = []
    seen: dict[tuple[str, str, str, str, str], Relationship] = {}

    for relationship in relationships:
        report.relationships_seen += 1
        normalized = _normalize_relationship(
            relationship,
            config=config,
            raw_to_normalized=raw_to_normalized,
            report=report,
        )
        if normalized is None:
            report.relationships_dropped += 1
            continue

        key = (
            str(normalized.source.id),
            str(normalized.source.type),
            str(normalized.type),
            str(normalized.target.id),
            str(normalized.target.type),
        )
        existing = seen.get(key)
        if existing is not None:
            existing.properties.update(
                _merge_properties(existing.properties, normalized.properties)
            )
            continue

        seen[key] = normalized
        normalized_relationships.append(normalized)
        report.relationships_kept += 1

    return normalized_relationships


def _normalize_relationship(
    relationship: Relationship,
    *,
    config: dict[str, Any],
    raw_to_normalized: dict[tuple[str, str], Node],
    report: NormalizationReport,
) -> Relationship | None:
    raw_type = str(relationship.type)
    type_name, rule = _resolve_relationship_type(raw_type, config)
    if rule is None and config.get("drop_unknown_relationships"):
        report.add_warning(f"Dropped relationship with unknown type: {raw_type}")
        return None

    source = raw_to_normalized.get(_node_key(relationship.source))
    target = raw_to_normalized.get(_node_key(relationship.target))
    if source is None or target is None:
        report.invalid_endpoint_count += 1
        report.add_warning(f"Dropped {type_name} relationship because an endpoint was dropped")
        return None

    expected_source = rule.get("source") if rule else None
    expected_target = rule.get("target") if rule else None
    if expected_source and expected_target:
        if source.type == expected_target and target.type == expected_source and rule.get("allow_reverse"):
            source, target = target, source
            report.relationships_reversed += 1
        elif source.type != expected_source or target.type != expected_target:
            report.invalid_endpoint_count += 1
            report.add_warning(
                f"Dropped {type_name} relationship with invalid endpoints: {source.type}->{target.type}"
            )
            return None

    properties = _normalize_properties(
        relationship.properties or {},
        rule=rule,
        config=config,
        section="relationships",
    )
    properties = _derive_properties(
        properties,
        rule=rule,
        context={
            "type": type_name,
            "source_id": str(source.id),
            "source_type": str(source.type),
            "target_id": str(target.id),
            "target_type": str(target.type),
            **{f"source_{key}": value for key, value in source.properties.items()},
            **{f"target_{key}": value for key, value in target.properties.items()},
            **properties,
        },
    )
    missing = _missing_required(properties, rule.get("required_properties") if rule else None)
    if missing:
        report.missing_required_property_count += 1
        report.add_warning(
            f"Dropped {type_name} relationship missing required properties: {', '.join(missing)}"
        )
        return None

    if config.get("preserve_raw", True) and type_name != raw_type:
        properties.setdefault("raw_type", raw_type)
        properties["normalization_actions"] = [f"type:{raw_type}->{type_name}"]
        report.relationships_renamed += 1

    return Relationship(
        source=source,
        target=target,
        type=type_name,
        properties=properties,
    )


def _resolve_node_type(raw_type: str, config: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    return _resolve_type(raw_type, config.get("nodes") or {})


def _resolve_relationship_type(
    raw_type: str,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    return _resolve_type(_canonical_relationship_type(raw_type), config.get("relationships") or {})


def _resolve_type(raw_type: str, rules: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if raw_type in rules:
        return raw_type, rules[raw_type] or {}

    normalized_raw = _type_token(raw_type)
    for canonical, rule in rules.items():
        aliases = [canonical, *_as_list((rule or {}).get("aliases"))]
        if normalized_raw in {_type_token(alias) for alias in aliases}:
            return canonical, rule or {}
    return raw_type, None


def _canonical_relationship_type(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").upper()


def _type_token(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _normalize_properties(
    properties: dict[str, Any],
    *,
    rule: dict[str, Any] | None,
    config: dict[str, Any],
    section: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    allowed = set(_as_list(rule.get("allowed_properties") if rule else None))
    property_rules = ((rule or {}).get("normalize_properties") or {})
    default_transforms = _as_list(config.get("default_transforms"))

    for key, value in properties.items():
        if value in (None, ""):
            continue
        if allowed and key not in allowed:
            continue
        normalized_value = _normalize_value(
            value,
            transforms=_as_list(property_rules.get(key)) or default_transforms,
        )
        if normalized_value in (None, ""):
            continue
        normalized[key] = normalized_value

    return normalized


def _derive_properties(
    properties: dict[str, Any],
    *,
    rule: dict[str, Any] | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    if not rule:
        return properties

    derived = dict(properties)
    allowed = set(_as_list(rule.get("allowed_properties")))
    derivation_rules = rule.get("derive_properties") or {}
    if not isinstance(derivation_rules, dict):
        return derived

    # Allow derivations to depend on earlier derived values.
    for _ in range(2):
        changed = False
        available = {**context, **derived}
        for key, spec in derivation_rules.items():
            if allowed and key not in allowed:
                continue
            if derived.get(key) not in (None, ""):
                continue

            value = _derive_property_value(spec, available)
            if value in (None, ""):
                continue
            derived[key] = value
            available[key] = value
            changed = True
        if not changed:
            break

    return derived


def _derive_property_value(spec: Any, context: dict[str, Any]) -> str | None:
    if isinstance(spec, str):
        return str(context.get(spec, "")) or None
    if not isinstance(spec, dict):
        return None

    if "value" in spec:
        value = spec["value"]
    elif "template" in spec:
        template = str(spec["template"])
        if not _template_fields_present(template, context):
            return None
        value = template.format(**context)
    else:
        source_key = spec.get("source")
        if source_key is None:
            return None
        value = context.get(str(source_key))

    if value in (None, ""):
        return None

    text = str(value)
    if spec.get("regex_extract"):
        text = _apply_transform(text, {"regex_extract": spec["regex_extract"]})
    for transform in _as_list(spec.get("transforms")):
        text = _apply_transform(text, transform)
    return text


def _build_node_id(
    raw_id: str,
    properties: dict[str, Any],
    rule: dict[str, Any] | None,
    config: dict[str, Any],
) -> str:
    if rule:
        id_property = rule.get("id_property")
        if id_property and properties.get(id_property) not in (None, ""):
            return str(properties[id_property])

        id_template = rule.get("id_template")
        if id_template and _template_fields_present(id_template, properties):
            return _render_template(id_template, properties, rule)

    return _normalize_value(raw_id, transforms=_as_list(config.get("default_transforms")))


def _template_fields_present(template: str, values: dict[str, Any]) -> bool:
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name and values.get(field_name) in (None, ""):
            return False
    return True


def _render_template(template: str, properties: dict[str, Any], rule: dict[str, Any]) -> str:
    template_values = dict(properties)
    lower_fields = set(_as_list(rule.get("lowercase_template_fields")))
    for field_name in lower_fields:
        if field_name in template_values:
            template_values[field_name] = str(template_values[field_name]).lower()
    return template.format(**template_values)


def _normalize_value(value: Any, *, transforms: list[Any]) -> str:
    text = _stringify_property(value)
    for transform in transforms:
        text = _apply_transform(text, transform)
    return text


def _stringify_property(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _apply_transform(value: str, transform: Any) -> str:
    if isinstance(transform, dict):
        if "regex_extract" in transform:
            match = re.search(str(transform["regex_extract"]), value)
            if not match:
                return ""
            if match.groupdict():
                return next(iter(match.groupdict().values()))
            return match.group(1) if match.groups() else match.group(0)
        if "regex_replace" in transform:
            spec = transform["regex_replace"] or {}
            return re.sub(str(spec.get("pattern", "")), str(spec.get("replacement", "")), value)
        return value

    name = str(transform)
    if name == "strip":
        return value.strip()
    if name == "collapse_whitespace":
        return re.sub(r"\s+", " ", value)
    if name == "fix_pdf_hyphenation":
        return re.sub(r"(\w)-\s+(\w)", r"\1\2", value)
    if name == "lowercase":
        return value.lower()
    if name == "uppercase":
        return value.upper()
    if name == "title_case":
        return value.title()
    if name == "colon_case":
        cleaned = re.sub(r"\s*:\s*", ":", value.strip())
        cleaned = re.sub(r"[\s_]+", ":", cleaned)
        return cleaned.lower()
    if name == "relationship_type":
        return _canonical_relationship_type(value)
    return value


def _missing_required(
    properties: dict[str, Any],
    required_properties: Any,
) -> list[str]:
    missing = []
    for key in _as_list(required_properties):
        if properties.get(key) in (None, ""):
            missing.append(str(key))
    return missing


def _matches_any(value: str, patterns: list[Any]) -> bool:
    return any(re.search(str(pattern), value) for pattern in patterns)


def _merge_properties(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key not in merged or merged[key] in (None, ""):
            merged[key] = value
    return merged


def _node_key(node: Node) -> tuple[str, str]:
    return (str(node.id), str(node.type))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
