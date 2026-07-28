from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    safe_mode = config.get("safe_mode", {})
    if safe_mode.get("dry_run") is not True:
        raise ConfigurationError("V1 exige dry_run=true")
    if safe_mode.get("publishing_enabled") is not False:
        raise ConfigurationError("V1 exige publishing_enabled=false")
    if safe_mode.get("social_tokens_allowed") is not False:
        raise ConfigurationError("V1 no permite tokens sociales")
    if sum(config.get("territories", {}).values()) != 100:
        raise ConfigurationError("La mezcla editorial debe sumar 100")
    weights = config.get("editorial_score", {}).get("weights", {})
    if sum(weights.values()) != 100:
        raise ConfigurationError("Los pesos editoriales deben sumar 100")
    selector = config.get("opportunity_selector", {})
    selector_weights = selector.get("weights", {})
    if sum(selector_weights.values()) != 100:
        raise ConfigurationError(
            "Los pesos del selector de oportunidades deben sumar 100"
        )
    minimum_score = selector.get("minimum_score")
    if not isinstance(minimum_score, (int, float)) or not (
        0 <= minimum_score <= 100
    ):
        raise ConfigurationError(
            "El mínimo del selector debe estar entre 0 y 100"
        )
    if selector.get("max_selected_per_run") != 1:
        raise ConfigurationError(
            "La prueba editorial solo puede seleccionar una oportunidad"
        )
    if selector.get("publishing_enabled") is not False:
        raise ConfigurationError(
            "El selector debe mantener publishing_enabled=false"
        )
    if selector.get("external_actions_enabled") is not False:
        raise ConfigurationError(
            "El selector debe mantener acciones externas apagadas"
        )
    if selector.get("human_approval_required") is not True:
        raise ConfigurationError(
            "El selector debe exigir aprobación humana"
        )
    if selector.get("views_only_success_allowed") is not False:
        raise ConfigurationError(
            "Las vistas no pueden ser el único criterio de éxito"
        )
    allowed_states = set(config.get("allowed_output_states", []))
    if not allowed_states or not allowed_states <= {"draft", "needs_review"}:
        raise ConfigurationError("V1 solo admite draft y needs_review")
