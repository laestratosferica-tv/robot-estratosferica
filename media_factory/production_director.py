from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_CONTENT_TYPES = {
    "noticia", "chisme", "cuento", "articulo", "dato", "relato",
    "polemica", "humor", "gameplay",
}


@dataclass(frozen=True)
class ProductionRequest:
    content_type: str
    title: str
    source_url: str
    factual_summary: str
    script: str
    character_id: str
    character_name: str
    editorial_voice: str
    language: str
    aspect_ratio: str
    duration_seconds: int
    heygen_group_id: str
    heygen_voice_id: str
    state: str
    blockers: list[str]
    production_enabled: bool = False
    external_actions_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cast(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_content_type(value: str) -> str:
    normalized = value.strip().lower().replace("í", "i").replace("ó", "o")
    if normalized not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(f"Tipo de contenido no soportado: {value}")
    return normalized


def select_character(content_type: str, cast: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized = normalize_content_type(content_type)
    for character_id, character in cast["characters"].items():
        if normalized in character["content_types"]:
            return character_id, character
    raise ValueError(f"No existe personaje para: {normalized}")


def build_production_request(
    payload: Mapping[str, Any], cast: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> ProductionRequest:
    env = os.environ if environment is None else environment
    content_type = normalize_content_type(str(payload.get("content_type", "")))
    character_id, character = select_character(content_type, cast)
    blockers: list[str] = []
    required = ("title", "source_url", "factual_summary", "script")
    for field in required:
        if not str(payload.get(field, "")).strip():
            blockers.append(f"missing_{field}")
    source_url = str(payload.get("source_url", "")).strip()
    if source_url and not source_url.startswith("https://"):
        blockers.append("source_must_use_https")
    group_id = str(env.get(character["heygen_group_env"], "")).strip()
    voice_id = str(env.get(character["heygen_voice_env"], "")).strip()
    if not group_id:
        blockers.append("missing_heygen_group_id")
    if not voice_id:
        blockers.append("missing_heygen_voice_id")
    if character["current_voice_status"] != "ready":
        blockers.append("voice_language_not_ready")
    duration = int(payload.get("duration_seconds", 25))
    if not 8 <= duration <= 60:
        blockers.append("duration_out_of_range")
    enabled = bool(cast.get("production_enabled", False)) and not blockers
    return ProductionRequest(
        content_type=content_type,
        title=str(payload.get("title", "")).strip(),
        source_url=source_url,
        factual_summary=str(payload.get("factual_summary", "")).strip(),
        script=str(payload.get("script", "")).strip(),
        character_id=character_id,
        character_name=character["display_name"],
        editorial_voice=character["editorial_voice"],
        language=cast["default_language"],
        aspect_ratio="9:16",
        duration_seconds=duration,
        heygen_group_id=group_id,
        heygen_voice_id=voice_id,
        state="ready_for_heygen" if enabled else "blocked",
        blockers=blockers,
        production_enabled=enabled,
        external_actions_enabled=False,
    )
