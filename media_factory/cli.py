from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .commercial import detect_opportunity
from .config import load_config
from .editor import evaluate_candidate
from .guardrails import validate_content_package, validate_storyboard
from .metrics import build_measurement_plan
from .models import Candidate, PipelineItem
from .queue import save_queue
from .radar import load_source_registry, normalize_story
from .studio import FORMAT_BY_TERRITORY, build_content_package
from .storyboard import build_storyboard
from .strategy import (
    classify_candidate,
    load_content_strategy,
    validate_strategy_decision,
)
from .talent import load_talent_catalog, select_talent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "config" / "content_strategy_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fábrica editorial V1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/editorial_queue.json")
    parser.add_argument("--sources")
    parser.add_argument(
        "--talent-config", default="config/talent_v1.json"
    )
    parser.add_argument(
        "--strategy-config", default=str(DEFAULT_STRATEGY)
    )
    return parser


def run_factory(
    config_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    sources_path: str | Path | None = None,
    talent_config_path: str | Path = "config/talent_v1.json",
    strategy_config_path: str | Path = DEFAULT_STRATEGY,
) -> dict[str, int]:
    config = load_config(config_path)
    talent_catalog = load_talent_catalog(talent_config_path)
    strategy = load_content_strategy(strategy_config_path)
    raw_candidates = json.loads(
        Path(input_path).read_text(encoding="utf-8")
    )
    limit = int(config["safe_mode"]["max_candidates_per_run"])
    if sources_path:
        registry = load_source_registry(sources_path)
        candidates = [
            normalize_story(item, registry) for item in raw_candidates[:limit]
        ]
    else:
        candidates = [
            Candidate.from_dict(item) for item in raw_candidates[:limit]
        ]
    candidates = [
        candidate
        if not validate_strategy_decision(
            candidate.strategic_classification
        )
        else replace(
            candidate,
            strategic_classification=classify_candidate(
                candidate,
                strategy,
            ),
        )
        for candidate in candidates
    ]
    decisions = [evaluate_candidate(item, config) for item in candidates]
    pipeline_items = []
    for candidate, decision in zip(candidates, decisions):
        opportunity = detect_opportunity(candidate, decision)
        talent = None
        if decision.accepted:
            format_id = FORMAT_BY_TERRITORY[candidate.territory]
            talent = select_talent(
                candidate.territory, format_id, talent_catalog
            ).to_dict()
        content_package = build_content_package(
            candidate, decision, opportunity, talent
        )
        if content_package:
            errors = validate_content_package(content_package)
            if errors:
                raise ValueError(
                    f"Paquete bloqueado por controles: {', '.join(errors)}"
                )
        storyboard = build_storyboard(candidate, content_package)
        if storyboard:
            errors = validate_storyboard(storyboard)
            if errors:
                raise ValueError(
                    f"Storyboard bloqueado por controles: {', '.join(errors)}"
                )
        pipeline_items.append(
            PipelineItem(
                candidate=candidate,
                decision=decision,
                commercial_opportunity=opportunity,
                content_package=content_package,
                storyboard=storyboard,
                measurement_plan=build_measurement_plan(decision, opportunity),
            )
        )
    accepted = [item for item in pipeline_items if item.decision.accepted]
    package_limit = int(config["safe_mode"]["max_packages_per_run"])
    rejected = [item for item in pipeline_items if not item.decision.accepted]
    save_queue(accepted[:package_limit] + rejected, output_path)
    return {
        "candidate_count": len(decisions),
        "strategy_classified_count": len(candidates),
        "accepted_count": len(accepted[:package_limit]),
        "rejected_count": len(rejected),
        "publication_count": 0,
    }


def main() -> int:
    args = build_parser().parse_args()
    summary = run_factory(
        args.config,
        args.input,
        args.output,
        sources_path=args.sources,
        talent_config_path=args.talent_config,
        strategy_config_path=args.strategy_config,
    )
    print(
        "Dry run completo: "
        f"{summary['candidate_count']} candidatos, 0 publicaciones"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
