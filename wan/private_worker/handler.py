#!/usr/bin/env python3
"""RunPod handler for the official Wan2.2 TI2V-5B implementation."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


LOG = logging.getLogger("estratosferica.wan")
ALLOWED_SIZES = {"1280*704", "704*1280"}
DEFAULT_NEGATIVE = (
    "identity drift, face distortion, deformed hands, extra fingers, "
    "wardrobe change, text, watermark, logo, low quality, blur"
)


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _decode_image(raw: Any, destination: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("image_base64 is required")
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[-1]
    max_bytes = int(os.getenv("WAN_MAX_IMAGE_BYTES", str(9 * 1024 * 1024)))
    if len(raw) > ((max_bytes + 2) // 3) * 4:
        raise ValueError("image_base64 exceeds the configured limit")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is invalid") from exc
    if len(decoded) > max_bytes:
        raise ValueError("decoded image exceeds the configured limit")
    if not (decoded.startswith(b"\x89PNG\r\n\x1a\n") or decoded.startswith(b"\xff\xd8\xff")):
        raise ValueError("image must be PNG or JPEG")
    destination.write_bytes(decoded)
    return destination


def validate_input(job_input: Any) -> dict[str, Any]:
    if not isinstance(job_input, dict):
        raise ValueError("input must be an object")
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    if len(prompt) > 4000:
        raise ValueError("prompt exceeds 4000 characters")
    image_base64 = job_input.get("image_base64")
    if not isinstance(image_base64, str) or not image_base64:
        raise ValueError("image_base64 is required")
    size = str(job_input.get("size", "704*1280"))
    if size not in ALLOWED_SIZES:
        raise ValueError(f"size must be one of {sorted(ALLOWED_SIZES)}")
    frame_num = _integer(job_input.get("frame_num", 121), "frame_num", 17, 121)
    if frame_num % 4 != 1:
        raise ValueError("frame_num must equal 4n+1")
    sample_steps = _integer(job_input.get("sample_steps", 30), "sample_steps", 10, 50)
    seed = _integer(job_input.get("seed", 42), "seed", 0, 2**31 - 1)
    return {
        "prompt": prompt.strip(),
        "negative_prompt": str(job_input.get("negative_prompt", DEFAULT_NEGATIVE))[:2000],
        "size": size,
        "frame_num": frame_num,
        "sample_steps": sample_steps,
        "seed": seed,
        "image_base64": image_base64,
    }


def build_command(validated: dict[str, Any], image_path: Path, output_path: Path) -> list[str]:
    source_dir = Path(os.getenv("WAN_SOURCE_DIR", "/opt/Wan2.2"))
    model_dir = Path(os.getenv("WAN_MODEL_DIR", "/runpod-volume/Wan2.2-TI2V-5B"))
    if not (source_dir / "generate.py").is_file():
        raise RuntimeError("official Wan source is missing")
    if not model_dir.is_dir():
        raise RuntimeError(
            "official Wan model is missing; run download_model.py on the mounted volume first"
        )
    prompt = f"{validated['prompt']}. Avoid: {validated['negative_prompt']}"
    return [
        "python", str(source_dir / "generate.py"),
        "--task", "ti2v-5B",
        "--size", validated["size"],
        "--ckpt_dir", str(model_dir),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--image", str(image_path),
        "--prompt", prompt,
        "--frame_num", str(validated["frame_num"]),
        "--sample_steps", str(validated["sample_steps"]),
        "--base_seed", str(validated["seed"]),
        "--save_file", str(output_path),
    ]


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    LOG.info("Received job input keys: %s", sorted(str(key) for key in job_input))
    validated = validate_input(job_input)
    job_id = str(job.get("id") or uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix=f"wan-{job_id[:24]}-") as temp_dir:
        temp = Path(temp_dir)
        image_path = _decode_image(validated["image_base64"], temp / "reference.png")
        output_path = temp / "result.mp4"
        command = build_command(validated, image_path, output_path)
        completed = subprocess.run(
            command,
            check=False,
            cwd=os.getenv("WAN_SOURCE_DIR", "/opt/Wan2.2"),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "")[-12000:]
            stdout_tail = (completed.stdout or "")[-4000:]
            raise RuntimeError(
                "Wan generation failed.\n"
                f"stdout tail:\n{stdout_tail}\n"
                f"stderr tail:\n{stderr_tail}"
            )
        if not output_path.is_file():
            raise RuntimeError("Wan completed without an output video")
        max_output = int(os.getenv("WAN_MAX_OUTPUT_BYTES", str(50 * 1024 * 1024)))
        if output_path.stat().st_size > max_output:
            raise RuntimeError("generated video exceeds the configured response limit")
        return {
            "video": base64.b64encode(output_path.read_bytes()).decode("ascii"),
            "mime_type": "video/mp4",
            "model": "Wan-AI/Wan2.2-TI2V-5B",
            "size": validated["size"],
            "frame_num": validated["frame_num"],
            "seed": validated["seed"],
        }


def self_test() -> int:
    sample = validate_input({
        "prompt": "Movimiento natural de presentadora",
        "image_base64": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(),
        "frame_num": 17,
        "sample_steps": 10,
    })
    assert sample["size"] == "704*1280"
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "input.png"
        _decode_image(sample["image_base64"], output)
        assert output.is_file()
    print(json.dumps({"safe": True, "model": "Wan-AI/Wan2.2-TI2V-5B"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    import runpod
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    runpod.serverless.start({"handler": handler})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
