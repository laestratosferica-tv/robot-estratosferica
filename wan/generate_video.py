from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from wan.client import ComfyClient, GenerationStats, ensure_mp4, load_json, prepare_workflow


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un clip Wan 2.2 TI2V-5B vía ComfyUI")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="identity drift, different face, different clothes, deformed")
    parser.add_argument("--output", type=Path, default=ROOT / "wan/outputs/wan-poc.mp4")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(ROOT / "wan/config/wan_poc.json")
    if not config.get("enabled"):
        raise SystemExit("POC bloqueada: cambia enabled=true solo cuando exista un endpoint autorizado.")
    if not config.get("allow_paid_remote"):
        raise SystemExit("Ejecución remota bloqueada: falta autorización explícita de gasto.")
    server_url = os.getenv("COMFYUI_URL") or config.get("server_url")
    if not server_url:
        raise SystemExit("Falta COMFYUI_URL.")
    if not args.image.is_file():
        raise SystemExit(f"No existe la imagen: {args.image}")

    defaults = config["defaults"]
    seed = defaults["seed"] if args.seed is None else args.seed
    client = ComfyClient(server_url, os.getenv("COMFYUI_TOKEN", ""))
    client.health()
    uploaded_name = client.upload_image(args.image)
    workflow = prepare_workflow(
        load_json(ROOT / config["workflow"]), uploaded_name, args.prompt, args.negative,
        defaults["width"], defaults["height"], defaults["frames"], defaults["fps"],
        defaults["steps"], defaults["cfg"], seed,
    )
    started = time.monotonic()
    prompt_id = client.queue_prompt(workflow)
    history, polls = client.wait_for_output(prompt_id)
    output_meta = client.find_output(history)
    with tempfile.TemporaryDirectory(prefix="wan-poc-") as temp_dir:
        source = client.download_output(output_meta, Path(temp_dir) / "result.bin")
        result = ensure_mp4(source, args.output)
    stats = GenerationStats(prompt_id, round(time.monotonic() - started, 2), polls, str(result))
    stats_path = args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(stats.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
