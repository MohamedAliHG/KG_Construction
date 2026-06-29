from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

import yaml

from config import settings

RelationshipSpec: TypeAlias = str | tuple[str, str, str]


class SchemaLevel(StrEnum):
    UNCONSTRAINED = "unconstrained"
    CONSTRAINED = "constrained"
    STRICT = "strict"


class ExtractionMode(StrEnum):
    TOOL = "tool"
    PROMPT = "prompt"


PropertySpec: TypeAlias = bool | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SchemaProfile:
    level: SchemaLevel
    allowed_nodes: tuple[str, ...] = ()
    allowed_relationships: tuple[RelationshipSpec, ...] = ()
    node_properties: PropertySpec = False
    relationship_properties: PropertySpec = False
    additional_instructions: str = None
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
        kwargs["additional_instructions"] = (
            self.additional_instructions
            if self.additional_instructions is not None
            else "")
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


def build_schema_profile(
    level: str | SchemaLevel | None,
    profile_path: str | Path | None = None,
) -> SchemaProfile:
    resolved = resolve_schema_level(level)

    if resolved == SchemaLevel.UNCONSTRAINED:
        return SchemaProfile(level=resolved)

    profile_data = load_schema_profile_data(profile_path or settings.schema_profile_path)
    allowed_nodes = tuple(profile_data.get("allowed_nodes") or ())
    relationships = profile_data.get("allowed_relationships") or {}

    if resolved == SchemaLevel.CONSTRAINED:
        return SchemaProfile(
            level=resolved,
            allowed_nodes=allowed_nodes,
            allowed_relationships=tuple(relationships.get("constrained") or ()),
        )

    if resolved == SchemaLevel.STRICT:
        return SchemaProfile(
            level=resolved,
            allowed_nodes=allowed_nodes,
            allowed_relationships=_parse_strict_relationships(
                relationships.get("strict") or ()
            ),
            additional_instructions=profile_data.get("additional_instructions") or "",
        )

    raise ValueError(f"Unsupported schema level: {level!r}")


def load_schema_profile_data(profile_path: str | Path | None) -> dict:
    if profile_path is None:
        raise ValueError("schema_profile_path is required for constrained or strict schema levels")

    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema profile file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Schema profile must be a mapping: {path}")
    return data


def _parse_strict_relationships(raw_relationships) -> tuple[RelationshipSpec, ...]:
    parsed = []
    for item in raw_relationships:
        if isinstance(item, str):
            parsed.append(item)
            continue
        if isinstance(item, dict):
            try:
                parsed.append((item["source"], item["type"], item["target"]))
            except KeyError as exc:
                raise ValueError(
                    "Strict relationship mappings require source, type, and target"
                ) from exc
            continue
        if isinstance(item, (list, tuple)) and len(item) == 3:
            parsed.append((item[0], item[1], item[2]))
            continue
        raise ValueError(f"Unsupported strict relationship entry: {item!r}")
    return tuple(parsed)
