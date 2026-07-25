from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commercial import detect_opportunity
from .config import load_config
from .editor import evaluate_candidate
from .metrics import build_measurement_plan
from .models import Candidate, PipelineItem
from .queue import save_queue
from .radar import load_source_registry, normalize_story


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fábrica editorial V1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/editorial_queue.json")
    parser.add_argument("--sources")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    raw_candidates = json.loads(Path(args.input).read_text(encoding="utf-8"))
    limit = int(config["safe_mode"]["max_candidates_per_run"])
    if args.sources:
        registry = load_source_registry(args.sources)
        candidates = [
            normalize_story(item, registry) for item in raw_candidates[:limit]
        ]
    else:
        candidates = [
            Candidate.from_dict(item) for item in raw_candidates[:limit]
        ]
    decisions = [evaluate_candidate(item, config) for item in candidates]
    pipeline_items = []
    for candidate, decision in zip(candidates, decisions):
        opportunity = detect_opportunity(candidate, decision)
        pipeline_items.append(
            PipelineItem(
                candidate=candidate,
                decision=decision,
                commercial_opportunity=opportunity,
                measurement_plan=build_measurement_plan(decision, opportunity),
            )
        )
    accepted = [item for item in pipeline_items if item.decision.accepted]
    package_limit = int(config["safe_mode"]["max_packages_per_run"])
    rejected = [item for item in pipeline_items if not item.decision.accepted]
    save_queue(accepted[:package_limit] + rejected, args.output)
    print(f"Dry run completo: {len(decisions)} candidatos, 0 publicaciones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
