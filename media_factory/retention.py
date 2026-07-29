from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Candidate


DEFAULT_PLAYBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "retention_playbook_v1.json"
)
REQUIRED_FORMATS = {
    "short_video",
    "photo",
    "carousel",
    "long_video",
    "text_post",
}
PROHIBITED_OBJECTIVES = {
    "adiccion",
    "compulsion",
    "enganio",
    "urgencia_falsa",
    "curiosidad_sin_recompensa",
    "explotacion_emocional",
}


def load_retention_playbook(
    path: Path = DEFAULT_PLAYBOOK_PATH,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_retention_playbook(playbook: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if playbook.get("schema_version") != "retention_playbook_v1":
        errors.append("invalid_retention_schema")

    identity = playbook.get("identity", {})
    forbidden_feeling = set(identity.get("forbidden_feeling", []))
    if "clase" not in forbidden_feeling or "tarea" not in forbidden_feeling:
        errors.append("missing_non_school_identity_rule")

    ethics = playbook.get("ethical_retention", {})
    prohibited = set(ethics.get("prohibited_objectives", []))
    if not PROHIBITED_OBJECTIVES <= prohibited:
        errors.append("incomplete_ethical_retention_policy")
    if ethics.get("objective") != "preferencia_recurrente_y_conexion":
        errors.append("invalid_retention_objective")

    evidence = playbook.get("evidence", [])
    if len(evidence) < 3:
        errors.append("insufficient_evidence_foundation")
    for source in evidence:
        if not source.get("url") or not source.get("operational_rule"):
            errors.append("incomplete_evidence_source")

    formats = playbook.get("formats", {})
    if set(formats) != REQUIRED_FORMATS:
        errors.append("incomplete_format_rules")
    for format_id, rules in formats.items():
        if not rules.get("retention_sequence"):
            errors.append(f"missing_retention_sequence:{format_id}")
        if not rules.get("visual_language"):
            errors.append(f"missing_visual_language:{format_id}")
        if not rules.get("metrics"):
            errors.append(f"missing_retention_metrics:{format_id}")

    protocol = playbook.get("experiment_protocol", {})
    if not protocol.get("baseline_required"):
        errors.append("retention_baseline_required")
    if not protocol.get("one_variable_per_version"):
        errors.append("single_variable_experiment_required")
    return sorted(set(errors))


def build_retention_plan(
    candidate: Candidate,
    format_id: str = "short_video",
    playbook: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = playbook or load_retention_playbook()
    errors = validate_retention_playbook(selected)
    if errors:
        raise ValueError(f"invalid retention playbook: {', '.join(errors)}")
    rules = selected["formats"][format_id]
    return {
        "schema_version": selected["schema_version"],
        "format": format_id,
        "audience": selected["identity"]["audience"],
        "relationship": selected["identity"]["relationship"],
        "story_territory": candidate.territory,
        "ethical_objective": selected["ethical_retention"]["objective"],
        "prohibited_objectives": selected["ethical_retention"][
            "prohibited_objectives"
        ],
        "universal_rules": selected["universal_rules"],
        "format_rules": rules,
        "evidence_ids": [item["id"] for item in selected["evidence"]],
        "experiment_protocol": selected["experiment_protocol"],
        "gate_passed": True,
    }


def validate_retention_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != "retention_playbook_v1":
        errors.append("missing_retention_plan")
    if plan.get("audience") != "gamers_latam_y_cultura_digital":
        errors.append("wrong_target_identity")
    if plan.get("relationship") != "persona_del_mismo_mundo_no_profesor":
        errors.append("school_like_target_relationship")
    if plan.get("ethical_objective") != "preferencia_recurrente_y_conexion":
        errors.append("unsafe_retention_objective")
    if not PROHIBITED_OBJECTIVES <= set(
        plan.get("prohibited_objectives", [])
    ):
        errors.append("missing_anti_manipulation_rules")
    if not plan.get("format_rules", {}).get("metrics"):
        errors.append("missing_format_retention_metrics")
    if not plan.get("experiment_protocol", {}).get(
        "one_variable_per_version"
    ):
        errors.append("invalid_retention_experiment")
    if not plan.get("gate_passed"):
        errors.append("retention_gate_failed")
    return sorted(set(errors))
