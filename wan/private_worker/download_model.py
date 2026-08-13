#!/usr/bin/env python3
"""Download the official Wan2.2 TI2V-5B model to a persistent volume."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"


def main() -> int:
    destination = Path(
        os.getenv("WAN_MODEL_DIR", "/runpod-volume/Wan2.2-TI2V-5B")
    ).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=destination,
        token=os.getenv("HF_TOKEN") or None,
    )
    required = (
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors.index.json",
    )
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"model download incomplete: {missing}")
    print(f"Official model ready at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
