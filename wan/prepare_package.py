from __future__ import annotations

import argparse
import json
from pathlib import Path

from wan.client import load_json, prepare_workflow


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara un paquete Wan sin usar GPU")
    parser.add_argument("--character", choices=("nova", "joseverso", "rami"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_package(character: str) -> dict:
    config = load_json(ROOT / "wan/config/wan_poc.json")
    characters = load_json(ROOT / "wan/config/characters.json")["characters"]
    profile = characters[character]
    reference = ROOT / profile["reference"]
    if not reference.is_file():
        raise SystemExit(f"Falta referencia privada: {reference}")
    defaults = config["defaults"]
    workflow = prepare_workflow(
        load_json(ROOT / config["workflow"]),
        reference.name,
        profile["prompt"],
        profile["negative"],
        defaults["width"], defaults["height"], defaults["frames"],
        defaults["fps"], defaults["steps"], defaults["cfg"], defaults["seed"],
    )
    return {
        "character": character,
        "role": profile["role"],
        "reference": str(reference),
        "reference_bytes": reference.stat().st_size,
        "workflow": workflow,
        "remote_execution": False,
        "authorization_required": "AUTORIZO RUNPOD",
    }


def main() -> int:
    args = parse_args()
    package = build_package(args.character)
    output = args.output or ROOT / f"wan/logs/{args.character}-prepared.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared_without_gpu", "character": args.character, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
