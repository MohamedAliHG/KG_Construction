from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

RelationshipSpec: TypeAlias = str | tuple[str, str, str]


class SchemaLevel(StrEnum):
    UNCONSTRAINED = "unconstrained"
    CONSTRAINED = "constrained"
    STRICT = "strict"


class ExtractionMode(StrEnum):
    TOOL = "tool"
    PROMPT = "prompt"


PropertySpec: TypeAlias = bool | tuple[str, ...]


DEFAULT_ALLOWED_NODES: tuple[str, ...] = (
    "Person",
    "Organization",
    "Location",
    "Award",
    "ResearchField",
)

DEFAULT_ALLOWED_RELATIONSHIPS: tuple[str, ...] = (
    "SPOUSE",
    "AWARD",
    "FIELD_OF_RESEARCH",
    "WORKS_AT",
    "IN_LOCATION",
)

DEFAULT_STRICT_RELATIONSHIPS: tuple[tuple[str, str, str], ...] = (
    ("Person", "SPOUSE", "Person"),
    ("Person", "AWARD", "Award"),
    ("Person", "WORKS_AT", "Organization"),
    ("Organization", "IN_LOCATION", "Location"),
    ("Person", "FIELD_OF_RESEARCH", "ResearchField"),
)


@dataclass(frozen=True, slots=True)
class SchemaProfile:
    level: SchemaLevel
    allowed_nodes: tuple[str, ...] = ()
    allowed_relationships: tuple[RelationshipSpec, ...] = ()
    node_properties: PropertySpec = False
    relationship_properties: PropertySpec = False
    strict_mode: bool = True

    def to_transformer_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"strict_mode": self.strict_mode}
        if self.allowed_nodes:
            kwargs["allowed_nodes"] = list(self.allowed_nodes)
        if self.allowed_relationships:
            kwargs["allowed_relationships"] = list(self.allowed_relationships)
        if self.node_properties:
            kwargs["node_properties"] = (
                list(self.node_properties)
                if isinstance(self.node_properties, tuple)
                else self.node_properties
            )
        if self.relationship_properties:
            kwargs["relationship_properties"] = (
                list(self.relationship_properties)
                if isinstance(self.relationship_properties, tuple)
                else self.relationship_properties
            )
        return kwargs


def parse_property_spec(value: str | bool | tuple[str, ...] | list[str] | None) -> PropertySpec:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)

    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in {"off", "false", "no", "0", ""}:
        return False
    if lowered in {"any", "true", "yes", "1"}:
        return True

    return tuple(part.strip() for part in normalized.split(",") if part.strip())


def resolve_schema_level(value: str | SchemaLevel | None) -> SchemaLevel:
    if value is None:
        return SchemaLevel.UNCONSTRAINED
    if isinstance(value, SchemaLevel):
        return value
    return SchemaLevel(value)


def resolve_extraction_mode(value: str | ExtractionMode | None) -> ExtractionMode:
    if value is None:
        return ExtractionMode.TOOL
    if isinstance(value, ExtractionMode):
        return value
    return ExtractionMode(value)


def build_schema_profile(level: str | SchemaLevel | None) -> SchemaProfile:
    resolved = resolve_schema_level(level)

    if resolved == SchemaLevel.UNCONSTRAINED:
        return SchemaProfile(level=resolved)

    if resolved == SchemaLevel.CONSTRAINED:
        return SchemaProfile(
            level=resolved,
            allowed_nodes=DEFAULT_ALLOWED_NODES,
            allowed_relationships=DEFAULT_ALLOWED_RELATIONSHIPS,
        )

    if resolved == SchemaLevel.STRICT:
        return SchemaProfile(
            level=resolved,
            allowed_nodes=DEFAULT_ALLOWED_NODES,
            allowed_relationships=DEFAULT_STRICT_RELATIONSHIPS,
        )

    raise ValueError(f"Unsupported schema level: {level!r}")
