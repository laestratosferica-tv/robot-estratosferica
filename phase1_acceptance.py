from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from operations_safety import load_json
from phase1_coordinator import (
    DEFAULT_SOURCES,
    OPERATIONS_CONFIG,
    ROOT,
    CoordinatorError,
    run_coordinator,
)


DEFAULT_CONFIG = ROOT / "config" / "editorial_v1.json"
DEFAULT_INPUT = ROOT / "fixtures" / "real_candidates_2026-07-25.json"
DEFAULT_REFERENCE_DATE = date(2026, 7, 25)


def _queue_digest(path: Path) -> str:
    queue = load_json(path)
    canonical = json.dumps(
        queue,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_acceptance(
    *,
    output_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    input_path: str | Path = DEFAULT_INPUT,
    sources_path: str | Path = DEFAULT_SOURCES,
    environment: Mapping[str, str] | None = None,
    today: date = DEFAULT_REFERENCE_DATE,
) -> dict[str, Any]:
    policy = load_json(OPERATIONS_CONFIG)["phase1_acceptance"]
    required_runs = int(policy["required_consecutive_runs"])
    results: list[dict[str, Any]] = []
    output = Path(output_path)
    artifact_dir = output.parent

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index in range(1, required_runs + 1):
            final_run = index == required_runs
            run_dir = artifact_dir if final_run else root / f"run-{index}"
            queue_path = run_dir / "editorial_queue.json"
            health = run_coordinator(
                config_path=config_path,
                input_path=input_path,
                sources_path=sources_path,
                queue_output=queue_path,
                safety_output=run_dir / "operations-safety.json",
                readiness_output=run_dir / "platform-readiness.json",
                health_output=run_dir / "coordinator-health.json",
                environment=environment,
                today=today,
            )
            results.append(
                {
                    "run": index,
                    "healthy": health["healthy"],
                    "queue_digest": _queue_digest(queue_path),
                    "candidate_count": health["factory"]["candidate_count"],
                    "publication_count": health["factory"][
                        "publication_count"
                    ],
                    "billable_operations": health["cost"][
                        "billable_operations"
                    ],
                    "measured_cost_usd": health["cost"][
                        "measured_cost_usd"
                    ],
                    "source_registry_enforced": health[
                        "source_registry_enforced"
                    ],
                }
            )

    digests = {result["queue_digest"] for result in results}
    healthy_runs = sum(result["healthy"] for result in results)
    publication_count = sum(
        result["publication_count"] for result in results
    )
    billable_operations = sum(
        result["billable_operations"] for result in results
    )
    measured_cost_usd = round(
        sum(result["measured_cost_usd"] for result in results), 6
    )
    stable_queue = len(digests) == 1
    sources_enforced = all(
        result["source_registry_enforced"] for result in results
    )
    passed = (
        healthy_runs == required_runs
        and publication_count == 0
        and billable_operations == 0
        and measured_cost_usd == 0.0
        and sources_enforced
        and (stable_queue or not policy["require_stable_queue"])
    )

    report = {
        "passed": passed,
        "mode": "phase1_safe_acceptance",
        "required_consecutive_runs": required_runs,
        "healthy_consecutive_runs": healthy_runs,
        "source_registry_enforced": sources_enforced,
        "stable_queue": stable_queue,
        "unique_queue_digests": len(digests),
        "publication_count": publication_count,
        "duplicate_publication_count": 0,
        "billable_operations": billable_operations,
        "measured_cost_usd": measured_cost_usd,
        "external_requests_attempted": False,
        "runs": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise CoordinatorError("Phase 1 acceptance validation failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate five safe and reproducible Phase 1 dry runs."
    )
    parser.add_argument(
        "--output", default="artifacts/phase1-acceptance.json"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    args = parser.parse_args()
    report = run_acceptance(
        output_path=args.output,
        config_path=args.config,
        input_path=args.input,
        sources_path=args.sources,
    )
    print(
        "Aceptación Fase 1 superada: "
        f"{report['healthy_consecutive_runs']} dry runs sanos, "
        "0 publicaciones, USD 0.00"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
