#!/usr/bin/env python3
"""Genera una prueba privada de Joseverso; nunca publica contenido."""

from __future__ import annotations

import argparse
from pathlib import Path

from media_factory.elevenlabs_tts import ElevenLabsTTS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-file", required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    script = Path(args.script_file).read_text(encoding="utf-8").strip()
    audio = ElevenLabsTTS().synthesize_approved_script(
        script, approval_id=args.approval_id
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio)
    print(f"Audio privado generado: {output.name} ({len(audio)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
