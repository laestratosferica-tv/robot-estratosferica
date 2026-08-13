from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from wan.client import load_json
from wan.runpod_client import RunpodGenerationStats, RunpodWanClient


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un clip en el worker privado Wan")
    parser.add_argument("--character", choices=("nova", "joseverso", "rami"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def build_payload(
    character: str,
    seed: int | None = None,
    reference_path: Path | None = None,
) -> tuple[dict, Path]:
    config = load_json(ROOT / "wan/config/wan_poc.json")
    profile = load_json(ROOT / "wan/config/characters.json")["characters"][character]
    reference = reference_path or (ROOT / profile["reference"])
    if not reference.is_file():
        raise SystemExit(f"Falta referencia privada: {reference}")
    defaults = config["runpod_defaults"]
    return {
        "image_base64": RunpodWanClient.encode_image(reference),
        "prompt": profile["prompt"],
        "negative_prompt": profile["negative"],
        "width": defaults["width"],
        "height": defaults["height"],
        "length": defaults["frames"],
        "steps": defaults["steps"],
        "cfg": defaults["cfg"],
        "seed": defaults["seed"] if seed is None else seed,
    }, reference


def main() -> int:
    args = parse_args()
    config = load_json(ROOT / "wan/config/wan_poc.json")
    if not config.get("enabled") or not config.get("allow_paid_remote"):
        raise SystemExit("RunPod bloqueado: habilita ambas compuertas solo para una ejecución aprobada.")
    if os.getenv("WAN_PRIVATE_WORKER") != "1":
        raise SystemExit("Worker bloqueado: WAN_PRIVATE_WORKER debe ser 1.")
    payload, _ = build_payload(args.character, args.seed)
    output = args.output or ROOT / f"wan/outputs/{args.character}-wan.mp4"
    client = RunpodWanClient(os.getenv("RUNPOD_ENDPOINT_ID", ""), os.getenv("RUNPOD_API_KEY", ""))
    started = time.monotonic()
    job_id = client.submit(payload)
    status, polls = client.wait(job_id)
    client.save_video(status, output)
    stats = RunpodGenerationStats(job_id, round(time.monotonic() - started, 2), polls, str(output), client.endpoint_id)
    output.with_suffix(".json").write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
