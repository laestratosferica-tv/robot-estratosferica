from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TalentSelection:
    character_id: str
    display_name: str
    role: str
    kind: str
    disclosure: str
    visual_identity_version: str
    voice_profile_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_talent_catalog(path: str | Path) -> dict[str, Any]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_public_catalog(catalog)
    return catalog


def validate_public_catalog(catalog: dict[str, Any]) -> None:
    forbidden = set(catalog.get("forbidden_public_fields", []))

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            leaked = forbidden.intersection(value)
            if leaked:
                raise ValueError(
                    "El catálogo público contiene campos privados: "
                    + ", ".join(sorted(leaked))
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(catalog)
    if not catalog.get("publication_requires_human_review", True):
        raise ValueError("El talento virtual V1 requiere revisión humana")


def select_talent(
    territory: str,
    format_id: str,
    catalog: dict[str, Any],
) -> TalentSelection:
    characters = {
        item["id"]: item
        for item in catalog.get("characters", [])
        if item.get("enabled") is True
    }
    preferred_ids = catalog.get("format_preference", {}).get(format_id, [])
    eligible = [
        characters[character_id]
        for character_id in preferred_ids
        if character_id in characters
        and territory in characters[character_id].get("territories", [])
        and format_id in characters[character_id].get("formats", [])
    ]
    if not eligible:
        eligible = [
            item
            for item in characters.values()
            if territory in item.get("territories", [])
            and format_id in item.get("formats", [])
        ]
        eligible.sort(key=lambda item: int(item.get("priority", 0)), reverse=True)
    if not eligible:
        raise ValueError(
            f"No hay talento habilitado para {territory}/{format_id}"
        )
    selected = eligible[0]
    return TalentSelection(
        character_id=selected["id"],
        display_name=selected["display_name"],
        role=selected["role"],
        kind=selected["kind"],
        disclosure=catalog["disclosure"],
        visual_identity_version=selected["visual_identity_version"],
        voice_profile_id=selected["voice_profile_id"],
    )
