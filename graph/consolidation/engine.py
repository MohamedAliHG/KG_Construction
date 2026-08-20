"""Config-driven graph entity and relationship consolidation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from config import settings
from graph.schema_profiles import load_schema_profile_data


@dataclass(slots=True)
class ConsolidationReport:
    graph_docs_seen: int = 0
    nodes_created: int = 0
    nodes_reused: int = 0
    relationships_created: int = 0
    duplicate_relationships_skipped: int = 0
    rules_matched: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        if len(self.warnings) < 50:
            self.warnings.append(message)


def consolidate_graph_documents(
    graph_docs: list[GraphDocument],
    *,
    schema_profile_path: str | None = None,
) -> tuple[list[GraphDocument], ConsolidationReport]:
    """Apply configured deterministic derivations to normalized graph documents."""
    profile_path = schema_profile_path or settings.schema_profile_path
    profile = load_schema_profile_data(profile_path)
    config = profile.get("consolidation") or {}
    if not isinstance(config, Mapping):
        raise ValueError("consolidation must be a mapping in the schema profile")

    report = ConsolidationReport(graph_docs_seen=len(graph_docs))
    if not config.get("enabled", False):
        return graph_docs, report

    _validate_rules_against_profile(config, profile)
    consolidated = [
        _consolidate_document(doc, config=config, report=report) for doc in graph_docs
    ]
    return consolidated, report


def _consolidate_document(
    graph_doc: GraphDocument,
    *,
    config: Mapping[str, Any],
    report: ConsolidationReport,
) -> GraphDocument:
    nodes = list(graph_doc.nodes)
    relationships = list(graph_doc.relationships)
    node_index = {_node_key(node): node for node in nodes}
    relationship_keys = {_relationship_key(rel) for rel in relationships}

    source_text = graph_doc.source.page_content if graph_doc.source else ""
    for rule in config.get("source_text_entity_rules") or ():
        _apply_source_text_entity_rule(
            rule,
            source_text=source_text,
            nodes=nodes,
            node_index=node_index,
            report=report,
        )

    for rule in config.get("reference_rules") or ():
        _apply_reference_rule(
            rule,
            nodes=nodes,
            node_index=node_index,
            relationships=relationships,
            relationship_keys=relationship_keys,
            report=report,
        )

    # Structural rules run after entity and reference rules so they also connect
    # newly derived Preparation and FlowchartBlockReference nodes.
    for rule in config.get("structural_rules") or ():
        _apply_structural_rule(
            rule,
            nodes=nodes,
            node_index=node_index,
            relationships=relationships,
            relationship_keys=relationship_keys,
            report=report,
        )

    for rule in config.get("cooccurrence_rules") or ():
        _apply_cooccurrence_rule(
            rule,
            source_text=graph_doc.source.page_content if graph_doc.source else "",
            nodes=nodes,
            relationships=relationships,
            relationship_keys=relationship_keys,
            report=report,
        )

    return GraphDocument(nodes=nodes, relationships=relationships, source=graph_doc.source)


def _apply_source_text_entity_rule(
    raw_rule: Any,
    *,
    source_text: str,
    nodes: list[Node],
    node_index: dict[tuple[str, str], Node],
    report: ConsolidationReport,
) -> None:
    """Create canonical entities from explicit source-text headings/captions."""
    rule = _rule_mapping(raw_rule, "source_text_entity")
    rule_id = str(rule.get("id") or "source_text_entity_rule")
    gate_pattern = rule.get("when_text_regex")
    if gate_pattern and re.search(
        str(gate_pattern), source_text, re.IGNORECASE | re.MULTILINE
    ) is None:
        return

    context, repeated = _extract_captures(
        source_text, rule.get("captures") or {}, rule_id
    )
    context["source_text"] = source_text
    matched = False
    for raw_output in rule.get("outputs") or ():
        output = _rule_mapping(raw_output, f"{rule_id}.output")
        repeated_name = output.get("for_each")
        contexts: Iterable[dict[str, Any]]
        if repeated_name:
            contexts = (
                {**context, **item}
                for item in repeated.get(str(repeated_name), ())
            )
        else:
            contexts = (context,)

        for output_context in contexts:
            required = [str(value) for value in (output.get("requires") or ())]
            if any(output_context.get(name) in (None, "") for name in required):
                continue
            target = _resolve_target(
                output.get("target"),
                context=output_context,
                nodes=nodes,
                node_index=node_index,
                report=report,
                rule_id=rule_id,
            )
            if target is not None:
                matched = True
    if matched:
        report.rules_matched += 1


def _apply_reference_rule(
    raw_rule: Any,
    *,
    nodes: list[Node],
    node_index: dict[tuple[str, str], Node],
    relationships: list[Relationship],
    relationship_keys: set[tuple[str, str, str, str, str]],
    report: ConsolidationReport,
) -> None:
    rule = _rule_mapping(raw_rule, "reference")
    rule_id = str(rule.get("id") or "reference_rule")
    source_type = _required_string(rule, "source_type", rule_id)
    source_property = str(rule.get("source_property") or "raw_text")

    for source in [node for node in list(nodes) if str(node.type) == source_type]:
        text = source.properties.get(source_property)
        if text in (None, ""):
            continue
        context, repeated = _extract_captures(str(text), rule.get("captures") or {}, rule_id)
        context[source_property] = str(text)
        context["source_id"] = str(source.id)
        matched = False
        for raw_output in rule.get("outputs") or ():
            output = _rule_mapping(raw_output, f"{rule_id}.output")
            repeated_name = output.get("for_each")
            contexts: Iterable[dict[str, Any]]
            if repeated_name:
                contexts = ({**context, **item} for item in repeated.get(str(repeated_name), ()))
            else:
                contexts = (context,)

            for output_context in contexts:
                required = [str(value) for value in (output.get("requires") or ())]
                if any(output_context.get(name) in (None, "") for name in required):
                    continue
                target = _resolve_target(
                    output.get("target"),
                    context=output_context,
                    nodes=nodes,
                    node_index=node_index,
                    report=report,
                    rule_id=rule_id,
                )
                if target is None:
                    continue
                _add_relationship(
                    source=source,
                    target=target,
                    spec=output.get("relationship"),
                    relationships=relationships,
                    relationship_keys=relationship_keys,
                    report=report,
                    rule_id=rule_id,
                    context=output_context,
                )
                matched = True
        if matched:
            report.rules_matched += 1


def _extract_captures(
    text: str,
    raw_captures: Any,
    rule_id: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not isinstance(raw_captures, Mapping):
        raise ValueError(f"Consolidation rule '{rule_id}' captures must be a mapping")
    context: dict[str, Any] = {}
    repeated: dict[str, list[dict[str, Any]]] = {}
    for name, raw_spec in raw_captures.items():
        spec = _rule_mapping(raw_spec, f"{rule_id}.captures.{name}")
        pattern = _required_string(spec, "regex", f"{rule_id}.captures.{name}")
        flags = re.IGNORECASE if spec.get("ignore_case", True) else 0
        if spec.get("find_all", False):
            repeated[str(name)] = [
                _clean_capture_values(match.groupdict())
                for match in re.finditer(pattern, text, flags)
            ]
            continue
        match = re.search(pattern, text, flags)
        if match:
            context.update(_clean_capture_values(match.groupdict()))
    return context, repeated


def _apply_structural_rule(
    raw_rule: Any,
    *,
    nodes: list[Node],
    node_index: dict[tuple[str, str], Node],
    relationships: list[Relationship],
    relationship_keys: set[tuple[str, str, str, str, str]],
    report: ConsolidationReport,
) -> None:
    rule = _rule_mapping(raw_rule, "structural")
    rule_id = str(rule.get("id") or "structural_rule")
    source_type = _required_string(rule, "source_type", rule_id)
    match_spec = _rule_mapping(rule.get("match"), f"{rule_id}.match")
    property_name = _required_string(match_spec, "property", f"{rule_id}.match")
    pattern = _required_string(match_spec, "regex", f"{rule_id}.match")

    for source in [node for node in list(nodes) if str(node.type) == source_type]:
        value = source.id if property_name == "id" else source.properties.get(property_name)
        if value in (None, ""):
            continue
        match = re.fullmatch(pattern, str(value), re.IGNORECASE)
        if not match:
            continue
        context = _clean_capture_values(match.groupdict())
        target = _resolve_target(
            rule.get("target"),
            context=context,
            nodes=nodes,
            node_index=node_index,
            report=report,
            rule_id=rule_id,
        )
        if target is None:
            continue
        _add_relationship(
            source=source,
            target=target,
            spec=rule.get("relationship"),
            relationships=relationships,
            relationship_keys=relationship_keys,
            report=report,
            rule_id=rule_id,
            context=context,
        )
        report.rules_matched += 1


def _apply_cooccurrence_rule(
    raw_rule: Any,
    *,
    source_text: str,
    nodes: list[Node],
    relationships: list[Relationship],
    relationship_keys: set[tuple[str, str, str, str, str]],
    report: ConsolidationReport,
) -> None:
    rule = _rule_mapping(raw_rule, "cooccurrence")
    rule_id = str(rule.get("id") or "cooccurrence_rule")
    pattern = rule.get("when_text_regex")
    if pattern and re.search(str(pattern), source_text, re.IGNORECASE | re.MULTILINE) is None:
        return
    source_type = _required_string(rule, "source_type", rule_id)
    target_type = _required_string(rule, "target_type", rule_id)
    sources = [node for node in nodes if str(node.type) == source_type]
    targets = [node for node in nodes if str(node.type) == target_type]
    if rule.get("require_exactly_one_source", True) and len(sources) != 1:
        return
    if rule.get("require_exactly_one_target", True) and len(targets) != 1:
        return
    if not sources or not targets:
        return
    for source in sources:
        for target in targets:
            _add_relationship(
                source=source,
                target=target,
                spec={"type": rule.get("relationship_type")},
                relationships=relationships,
                relationship_keys=relationship_keys,
                report=report,
                rule_id=rule_id,
                context={},
            )
    report.rules_matched += 1


def _resolve_target(
    raw_spec: Any,
    *,
    context: Mapping[str, Any],
    nodes: list[Node],
    node_index: dict[tuple[str, str], Node],
    report: ConsolidationReport,
    rule_id: str,
) -> Node | None:
    spec = _rule_mapping(raw_spec, f"{rule_id}.target")
    target_type = _required_string(spec, "type", f"{rule_id}.target")
    target_id = _render_template(
        _required_string(spec, "id_template", f"{rule_id}.target"), context
    )
    if not target_id:
        report.add_warning(f"Rule '{rule_id}' produced an empty target id")
        return None
    key = (target_type, target_id)
    existing = node_index.get(key)
    if existing is not None:
        report.nodes_reused += 1
        return existing
    properties = _render_value(spec.get("properties") or {}, context)
    target = Node(id=target_id, type=target_type, properties=properties)
    nodes.append(target)
    node_index[key] = target
    report.nodes_created += 1
    return target


def _add_relationship(
    *,
    source: Node,
    target: Node,
    spec: Any,
    relationships: list[Relationship],
    relationship_keys: set[tuple[str, str, str, str, str]],
    report: ConsolidationReport,
    rule_id: str,
    context: Mapping[str, Any],
) -> None:
    relationship_spec = _rule_mapping(spec, f"{rule_id}.relationship")
    relationship_type = _required_string(
        relationship_spec, "type", f"{rule_id}.relationship"
    )
    direction = str(relationship_spec.get("direction") or "source_to_target")
    if direction == "target_to_source":
        rel_source, rel_target = target, source
    elif direction == "source_to_target":
        rel_source, rel_target = source, target
    else:
        raise ValueError(f"Rule '{rule_id}' has unsupported direction '{direction}'")
    key = (
        str(rel_source.type),
        str(rel_source.id),
        relationship_type,
        str(rel_target.type),
        str(rel_target.id),
    )
    if key in relationship_keys:
        report.duplicate_relationships_skipped += 1
        return
    properties = _render_value(relationship_spec.get("properties") or {}, context)
    properties.update({"derived": True, "derivation_rule": rule_id})
    relationships.append(
        Relationship(
            source=rel_source,
            target=rel_target,
            type=relationship_type,
            properties=properties,
        )
    )
    relationship_keys.add(key)
    report.relationships_created += 1

def _validate_rules_against_profile(
    config: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    allowed_nodes = {str(value) for value in (profile.get("allowed_nodes") or ())}
    strict_specs = (profile.get("allowed_relationships") or {}).get("strict") or ()
    allowed_relationships = {
        (str(item.get("source")), str(item.get("type")), str(item.get("target")))
        for item in strict_specs
        if isinstance(item, Mapping)
    }
    if not allowed_nodes and not allowed_relationships:
        return
    for section in (
        "source_text_entity_rules",
        "structural_rules",
        "reference_rules",
        "cooccurrence_rules",
    ):
        for raw_rule in config.get(section) or ():
            rule = _rule_mapping(raw_rule, section)
            source_type = rule.get("source_type")
            if source_type and allowed_nodes and str(source_type) not in allowed_nodes:
                raise ValueError(
                    f"Consolidation rule source type '{source_type}' is not allowed"
                )
            if section == "source_text_entity_rules":
                for output in rule.get("outputs") or ():
                    output = _rule_mapping(output, "output")
                    target = _rule_mapping(output.get("target"), "target")
                    target_type = target.get("type")
                    if (
                        target_type
                        and allowed_nodes
                        and str(target_type) not in allowed_nodes
                    ):
                        raise ValueError(
                            f"Source-text target type {target_type!r} is not allowed"
                        )
                specs = []
            elif section == "cooccurrence_rules":
                specs = [
                    (source_type, rule.get("relationship_type"), rule.get("target_type"))
                ]
            elif section == "structural_rules":
                rel = _rule_mapping(rule.get("relationship"), "relationship")
                target = _rule_mapping(rule.get("target"), "target")
                if rel.get("direction") == "target_to_source":
                    specs = [(target.get("type"), rel.get("type"), source_type)]
                else:
                    specs = [(source_type, rel.get("type"), target.get("type"))]
            else:
                specs = []
                for output in rule.get("outputs") or ():
                    output = _rule_mapping(output, "output")
                    target = _rule_mapping(output.get("target"), "target")
                    rel = _rule_mapping(output.get("relationship"), "relationship")
                    specs.append((source_type, rel.get("type"), target.get("type")))
            for spec in specs:
                normalized = tuple(str(value) for value in spec)
                if allowed_relationships and normalized not in allowed_relationships:
                    raise ValueError(
                        f"Derived relationship {normalized} is not allowed by strict schema"
                    )


def _render_template(template: str, context: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        expression = match.group(1)
        parts = expression.split("|")
        value = context.get(parts[0])
        if value in (None, ""):
            return ""
        rendered = str(value)
        for transform in parts[1:]:
            if transform == "lower":
                rendered = rendered.lower()
            elif transform == "upper":
                rendered = rendered.upper()
            else:
                raise ValueError(
                    f"Unsupported consolidation template transform: {transform}"
                )
        return rendered

    pattern = r"\{([A-Za-z_][A-Za-z0-9_]*(?:\|[A-Za-z_]+)*)\}"
    return re.sub(pattern, replace, template)


def _render_value(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _render_template(value, context)
    if isinstance(value, Mapping):
        return {key: _render_value(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_value(item, context) for item in value]
    return value


def _clean_capture_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.strip() if isinstance(value, str) else value
        for key, value in values.items()
        if value is not None
    }


def _node_key(node: Node) -> tuple[str, str]:
    return str(node.type), str(node.id)


def _relationship_key(rel: Relationship) -> tuple[str, str, str, str, str]:
    return (
        str(rel.source.type),
        str(rel.source.id),
        str(rel.type),
        str(rel.target.type),
        str(rel.target.id),
    )


def _rule_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Consolidation {location} must be a mapping")
    return dict(value)


def _required_string(mapping: Mapping[str, Any], key: str, location: str) -> str:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Consolidation {location} requires '{key}'")
    return str(value)
