from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Candidate


DEFAULT_VARIATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "creative_variation_v1.json"
)
REQUIRED_VOICE_PRESENTATIONS = {"neutral_robot", "masculine", "feminine"}


def load_creative_variation(
    path: Path = DEFAULT_VARIATION_PATH,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_creative_variation(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != "creative_variation_v1":
        errors.append("invalid_creative_variation_schema")
    voices = config.get("voice_cast", [])
    presentations = {voice.get("presentation") for voice in voices}
    if not REQUIRED_VOICE_PRESENTATIONS <= presentations:
        errors.append("incomplete_voice_cast")
    if len({world.get("id") for world in config.get("worlds", [])}) < 6:
        errors.append("insufficient_multiverse_worlds")
    for key in ("hook_patterns", "motion_systems", "text_behaviors"):
        if len(set(config.get(key, []))) < 4:
            errors.append(f"insufficient_creative_options:{key}")
    rotation = config.get("rotation_rules", {})
    for key in (
        "max_consecutive_same_voice",
        "max_consecutive_same_world",
        "max_consecutive_same_hook",
        "max_consecutive_same_motion",
    ):
        if rotation.get(key) != 1:
            errors.append(f"repetition_not_blocked:{key}")
    if not rotation.get("content_relevance_over_randomness"):
        errors.append("content_relevance_must_win")
    return sorted(set(errors))


def _index(seed: str, namespace: str, size: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}".encode()).hexdigest()
    return int(digest[:12], 16) % size


def _without_previous(
    options: list[dict[str, Any]] | list[str],
    previous: str,
) -> list[dict[str, Any]] | list[str]:
    filtered = [
        option
        for option in options
        if (option.get("id") if isinstance(option, dict) else option)
        != previous
    ]
    return filtered or options


def _choose(
    options: list[dict[str, Any]] | list[str],
    seed: str,
    namespace: str,
    previous: str = "",
) -> dict[str, Any] | str:
    available = _without_previous(options, previous)
    return available[_index(seed, namespace, len(available))]


def select_creative_profile(
    candidate: Candidate,
    history: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = config or load_creative_variation()
    errors = validate_creative_variation(selected)
    if errors:
        raise ValueError(f"invalid creative variation: {', '.join(errors)}")

    recent = history or []
    previous = recent[-1] if recent else {}
    seed = candidate.candidate_id or f"{candidate.title}:{candidate.source_url}"

    voices = selected["voice_cast"]
    if not recent:
        voice = next(
            item
            for item in voices
            if item["id"] == selected["rotation_rules"]["first_voice"]
        )
    else:
        relevant_voices = [
            item
            for item in voices
            if candidate.territory in item.get("best_for", [])
        ] or voices
        voice = _choose(
            relevant_voices,
            seed,
            "voice",
            str(previous.get("voice_id", "")),
        )

    relevant_worlds = [
        world
        for world in selected["worlds"]
        if candidate.territory in world["territories"]
    ] or selected["worlds"]
    world = _choose(
        relevant_worlds,
        seed,
        "world",
        str(previous.get("world_id", "")),
    )
    hook = _choose(
        selected["hook_patterns"],
        seed,
        "hook",
        str(previous.get("hook_pattern", "")),
    )
    motion = _choose(
        selected["motion_systems"],
        seed,
        "motion",
        str(previous.get("motion_system", "")),
    )
    text = _choose(
        selected["text_behaviors"],
        seed,
        "text",
        str(previous.get("text_behavior", "")),
    )

    return {
        "schema_version": selected["schema_version"],
        "voice_id": voice["id"],
        "voice_presentation": voice["presentation"],
        "voice_energy": voice["energy"],
        "world_id": world["id"],
        "visual_grammar": world["visual_grammar"],
        "hook_pattern": hook,
        "motion_system": motion,
        "text_behavior": text,
        "invariants": selected["invariants"],
        "selection_reason": "territory_fit_plus_no_immediate_repetition",
        "gate_passed": True,
    }


def validate_creative_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if profile.get("schema_version") != "creative_variation_v1":
        errors.append("missing_creative_profile")
    if profile.get("voice_presentation") not in REQUIRED_VOICE_PRESENTATIONS:
        errors.append("invalid_voice_presentation")
    for field in (
        "voice_id",
        "world_id",
        "hook_pattern",
        "motion_system",
        "text_behavior",
        "visual_grammar",
    ):
        if not profile.get(field):
            errors.append(f"missing_creative_dimension:{field}")
    if "no_imitar_personas_reales" not in profile.get("invariants", []):
        errors.append("missing_voice_identity_safety")
    if not profile.get("gate_passed"):
        errors.append("creative_variation_gate_failed")
    return sorted(set(errors))
