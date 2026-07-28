from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from media_factory.cli import run_factory
from media_factory.strategy import validate_strategy_decision
from operations_safety import build_safety_report, load_json


ROOT = Path(__file__).resolve().parent
OPERATIONS_CONFIG = ROOT / "config" / "operations_v1.json"
DEFAULT_SOURCES = ROOT / "config" / "sources_v1.json"


class CoordinatorError(RuntimeError):
    pass


def _configured_flag(name: str, environment: Mapping[str, str]) -> bool | None:
    raw = environment.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise CoordinatorError(f"Invalid credential presence flag: {name}")


def build_platform_readiness(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    config = load_json(OPERATIONS_CONFIG)["platform_readiness"]
    platforms: dict[str, Any] = {}

    for platform, required_flags in config.items():
        presence = {
            flag: _configured_flag(flag, environment)
            for flag in required_flags
        }
        known = [value for value in presence.values() if value is not None]
        if not known:
            status = "unknown"
        elif len(known) != len(presence):
            status = "partial_visibility"
        elif all(known):
            status = "configured_not_validated"
        elif any(known):
            status = "incomplete"
        else:
            status = "missing"
        platforms[platform] = {
            "status": status,
            "required_count": len(required_flags),
            "configured_count": sum(value is True for value in presence.values()),
            "secret_values_exposed": False,
            "external_validation_performed": False,
        }

    return {
        "mode": "presence_only",
        "publishing_attempted": False,
        "external_requests_attempted": False,
        "platforms": platforms,
    }


def _validate_review_queue(path: Path) -> dict[str, Any]:
    queue = load_json(path)
    errors: list[str] = []
    if queue.get("mode") != "dry_run":
        errors.append("review queue must remain in dry_run")
    if queue.get("publishing_enabled") is not False:
        errors.append("review queue publishing must be disabled")
    if queue.get("external_actions_enabled") is not False:
        errors.append("review queue external actions must be disabled")
    if queue.get("schema_version") != "review_queue_v1":
        errors.append("review queue schema must be review_queue_v1")
    if queue.get("human_approval_required") is not True:
        errors.append("review queue must require human approval")
    selection_report = queue.get("opportunity_selection", {})
    if selection_report.get("schema_version") != "opportunity_selection_v1":
        errors.append("review queue must include opportunity selection v1")
    if selection_report.get("selected_count") not in {0, 1}:
        errors.append("selector can choose at most one opportunity")
    if selection_report.get("views_only_success_allowed") is not False:
        errors.append("views cannot be the only success criterion")
    if selection_report.get("publishing_enabled") is not False:
        errors.append("selector publishing must remain disabled")
    if selection_report.get("external_actions_enabled") is not False:
        errors.append("selector external actions must remain disabled")
    for item in queue.get("items", []):
        review = item.get("review", {})
        if review.get("status") != "pending_human_approval":
            errors.append("review item must remain pending human approval")
        if review.get("publish_allowed") is not False:
            errors.append("review item cannot allow publishing")
        for strategy_error in validate_strategy_decision(
            review.get("strategy", {})
        ):
            errors.append(f"review strategy: {strategy_error}")
        selection = review.get("opportunity_selection", {})
        if selection.get("selected") is not True:
            errors.append("review item must be selected by the selector")
        if selection.get("eligible") is not True:
            errors.append("review item must be selector eligible")
        editorial_test = review.get("editorial_test", {})
        for field in (
            "objective",
            "expected_interaction",
            "interaction_prompt",
            "primary_metric",
            "audience_hypothesis",
        ):
            if not editorial_test.get(field):
                errors.append(f"editorial test missing {field}")
        if editorial_test.get("state") != "draft":
            errors.append("editorial test must remain draft")
        if editorial_test.get("views_only_success_allowed") is not False:
            errors.append("editorial test cannot use views-only success")
        if editorial_test.get("publishing_enabled") is not False:
            errors.append("editorial test cannot publish")
        if editorial_test.get("external_actions_enabled") is not False:
            errors.append("editorial test cannot perform external actions")
    if selection_report.get("selected_count") != len(queue.get("items", [])):
        errors.append("selection count must match review queue")
    return {
        "safe": not errors,
        "item_count": len(queue.get("items", [])),
        "publishing_enabled": queue.get("publishing_enabled"),
        "external_actions_enabled": queue.get("external_actions_enabled"),
        "errors": errors,
    }


def run_coordinator(
    *,
    config_path: str | Path,
    input_path: str | Path,
    queue_output: str | Path,
    safety_output: str | Path,
    readiness_output: str | Path,
    health_output: str | Path,
    sources_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    safety = build_safety_report()
    if not safety["safe"]:
        raise CoordinatorError("Safety baseline is not green")

    acceptance = load_json(OPERATIONS_CONFIG)["phase1_acceptance"]
    if acceptance["require_source_registry"] and sources_path is None:
        raise CoordinatorError("Phase 1 requires the verified source registry")

    factory = run_factory(
        config_path,
        input_path,
        queue_output,
        sources_path=sources_path,
    )
    queue = _validate_review_queue(Path(queue_output))
    readiness = build_platform_readiness(environment)
    billable_operations = 0
    measured_cost_usd = 0.0
    cost_within_limit = (
        billable_operations
        <= acceptance["max_billable_operations_per_run"]
        and measured_cost_usd <= acceptance["max_cost_usd_per_run"]
    )
    cost = {
        "currency": "USD",
        "billable_operations": billable_operations,
        "billable_operation_limit": acceptance[
            "max_billable_operations_per_run"
        ],
        "measured_cost_usd": measured_cost_usd,
        "limit_usd_per_run": acceptance["max_cost_usd_per_run"],
        "within_limit": cost_within_limit,
        "measurement_basis": (
            "No external requests or paid generation were attempted"
        ),
    }

    health = {
        "healthy": safety["safe"] and queue["safe"] and cost["within_limit"],
        "mode": "manual_safe_dry_run",
        "coordinator": "phase1_coordinator.py",
        "safety": safety,
        "factory": factory,
        "review_queue": queue,
        "platform_readiness": readiness,
        "source_registry_enforced": True,
        "cost": cost,
        "publishing_attempted": False,
        "external_writes_attempted": False,
        "paid_generation_attempted": False,
    }

    outputs = (
        (Path(safety_output), safety),
        (Path(readiness_output), readiness),
        (Path(health_output), health),
    )
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if not health["healthy"]:
        raise CoordinatorError("Coordinator health validation failed")
    return health


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Estratosferica 3.0 Phase 1 in safe manual mode."
    )
    parser.add_argument("--config", default="config/editorial_v1.json")
    parser.add_argument(
        "--input", default="fixtures/real_candidates_2026-07-25.json"
    )
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument(
        "--queue-output", default="artifacts/editorial_queue.json"
    )
    parser.add_argument(
        "--safety-output", default="artifacts/operations-safety.json"
    )
    parser.add_argument(
        "--readiness-output", default="artifacts/platform-readiness.json"
    )
    parser.add_argument(
        "--health-output", default="artifacts/coordinator-health.json"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    health = run_coordinator(
        config_path=args.config,
        input_path=args.input,
        sources_path=args.sources,
        queue_output=args.queue_output,
        safety_output=args.safety_output,
        readiness_output=args.readiness_output,
        health_output=args.health_output,
    )
    print(
        "Coordinador Fase 1 saludable: "
        f"{health['factory']['candidate_count']} candidatos, "
        "0 publicaciones, 0 escrituras externas, 0 generación paga"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
