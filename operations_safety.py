from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / ".github" / "workflows"
OPERATIONS_CONFIG = ROOT / "config" / "operations_v1.json"
EDITORIAL_CONFIG = ROOT / "config" / "editorial_v1.json"
ACCOUNTS_CONFIG = ROOT / "accounts.json"


class SafetyError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow_files() -> set[str]:
    return {path.name for path in WORKFLOWS.glob("*.yml")}


def build_safety_report() -> dict[str, Any]:
    operations = load_json(OPERATIONS_CONFIG)
    defaults = operations["defaults"]
    inventory = operations["workflow_inventory"]
    actual_workflows = _workflow_files()
    expected_workflows = set(inventory)

    errors: list[str] = []
    if actual_workflows != expected_workflows:
        missing = sorted(actual_workflows - expected_workflows)
        stale = sorted(expected_workflows - actual_workflows)
        errors.append(
            f"workflow inventory mismatch; missing={missing}, stale={stale}"
        )

    unsafe_defaults = {
        "scheduled_runs_enabled": False,
        "external_writes_enabled": False,
        "publishing_enabled": False,
        "paid_generation_enabled": False,
        "dry_run": True,
    }
    for key, expected in unsafe_defaults.items():
        if defaults.get(key) is not expected:
            errors.append(f"unsafe operations default: {key}={defaults.get(key)!r}")

    legacy_gate = operations["activation_rules"]["legacy_gate"]
    production_gate = operations["activation_rules"]["production_gate"]
    quarantined = sorted(
        filename
        for filename, role in inventory.items()
        if role == "legacy_quarantine"
    )
    scheduled: list[str] = []
    for filename in sorted(actual_workflows):
        content = (WORKFLOWS / filename).read_text(encoding="utf-8")
        if "schedule:" in content:
            scheduled.append(filename)
        if filename in quarantined and legacy_gate not in content:
            errors.append(f"{filename} is missing the legacy quarantine gate")

    for filename in operations["legacy_publishers"]:
        content = (WORKFLOWS / filename).read_text(encoding="utf-8")
        if production_gate not in content:
            errors.append(f"{filename} is missing the production gate")

    allowed_publish_scheduled = sorted(
        filename
        for filename, role in inventory.items()
        if role == "controlled_scheduled_publisher"
    )
    allowed_validation_scheduled = sorted(
        filename
        for filename, role in inventory.items()
        if role == "controlled_scheduled_validator"
    )
    allowed_scheduled = sorted(
        set(allowed_publish_scheduled) | set(allowed_validation_scheduled)
    )
    unexpected_scheduled = sorted(set(scheduled) - set(allowed_scheduled))
    if unexpected_scheduled:
        errors.append(f"unexpected scheduled workflows enabled: {unexpected_scheduled}")
    for filename in allowed_publish_scheduled:
        content = (WORKFLOWS / filename).read_text(encoding="utf-8")
        if "schedule:" not in content:
            errors.append(f"{filename} is missing its controlled schedule")
        if production_gate not in content:
            errors.append(f"{filename} is missing the production gate")
        if "vars.SCHEDULED_PUBLISHING_ARMED == 'true'" not in content:
            errors.append(f"{filename} is missing the scheduled publishing gate")
    for filename in allowed_validation_scheduled:
        content = (WORKFLOWS / filename).read_text(encoding="utf-8")
        if "schedule:" not in content:
            errors.append(f"{filename} is missing its controlled schedule")
        if "contents: read" not in content:
            errors.append(f"{filename} must remain read-only")
        if "deploy_site_version" in content or "git push" in content:
            errors.append(f"{filename} cannot deploy or push")

    editorial = load_json(EDITORIAL_CONFIG)["safe_mode"]
    if editorial.get("dry_run") is not True:
        errors.append("editorial safe_mode.dry_run must be true")
    if editorial.get("publishing_enabled") is not False:
        errors.append("editorial publishing must be disabled")
    if editorial.get("social_tokens_allowed") is not False:
        errors.append("social tokens must be blocked in Factory V1")
    if editorial.get("runway_enabled") is not False:
        errors.append("Runway must be disabled in Factory V1")

    for account in load_json(ACCOUNTS_CONFIG)["accounts"]:
        threads = account.get("threads", {})
        if threads.get("auto_post") is not False:
            errors.append(f"{account['account_id']}: Threads auto_post must be false")
        if threads.get("auto_post_limit") != 0:
            errors.append(
                f"{account['account_id']}: Threads auto_post_limit must be zero"
            )
        if threads.get("dry_run") is not True:
            errors.append(f"{account['account_id']}: Threads dry_run must be true")

    return {
        "safe": not errors,
        "mode": operations["mode"],
        "coordinator": operations["coordinator_workflow"],
        "workflow_count": len(actual_workflows),
        "quarantined_workflow_count": len(quarantined),
        "scheduled_workflows": scheduled,
        "publishing_enabled": defaults["publishing_enabled"],
        "external_writes_enabled": defaults["external_writes_enabled"],
        "paid_generation_enabled": defaults["paid_generation_enabled"],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Estratosferica 3.0 production safety baseline."
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_safety_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["safe"]:
        raise SafetyError("Production safety baseline validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
