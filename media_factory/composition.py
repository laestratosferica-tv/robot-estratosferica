from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CompositionPlan:
    content_id: str
    presenter_video: str
    context_videos: list[str]
    wan_scenes: list[str]
    graphics_manifest: str
    captions_file: str
    source_credit: str
    output_file: str
    state: str
    blockers: list[str]
    render_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_composition_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_composition_plan(payload: Mapping[str, Any], config: Mapping[str, Any]) -> CompositionPlan:
    blockers: list[str] = []
    presenter = str(payload.get("presenter_video", "")).strip()
    captions = str(payload.get("captions_file", "")).strip()
    credit = str(payload.get("source_credit", "")).strip()
    output = str(payload.get("output_file", "")).strip()
    if not presenter:
        blockers.append("missing_heygen_presenter")
    if not captions:
        blockers.append("missing_captions")
    if config["qa"]["require_source_credit"] and not credit:
        blockers.append("missing_source_credit")
    if not output.endswith(".mp4"):
        blockers.append("output_must_be_mp4")
    graphics = str(payload.get("graphics_manifest", "")).strip()
    state = "ready_for_render" if not blockers else "blocked"
    return CompositionPlan(
        content_id=str(payload.get("content_id", "")).strip(),
        presenter_video=presenter,
        context_videos=list(payload.get("context_videos", [])),
        wan_scenes=list(payload.get("wan_scenes", [])),
        graphics_manifest=graphics,
        captions_file=captions,
        source_credit=credit,
        output_file=output,
        state=state,
        blockers=blockers,
        render_enabled=bool(config.get("production_enabled", False)) and not blockers,
    )
