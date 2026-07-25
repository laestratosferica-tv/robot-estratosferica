from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .editor import evaluate_candidate
from .models import Candidate
from .queue import save_queue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fábrica editorial V1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/editorial_queue.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    raw_candidates = json.loads(Path(args.input).read_text(encoding="utf-8"))
    limit = int(config["safe_mode"]["max_candidates_per_run"])
    candidates = [Candidate.from_dict(item) for item in raw_candidates[:limit]]
    decisions = [evaluate_candidate(item, config) for item in candidates]
    accepted = [item for item in decisions if item.accepted]
    package_limit = int(config["safe_mode"]["max_packages_per_run"])
    rejected = [item for item in decisions if not item.accepted]
    save_queue(accepted[:package_limit] + rejected, args.output)
    print(f"Dry run completo: {len(decisions)} candidatos, 0 publicaciones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
