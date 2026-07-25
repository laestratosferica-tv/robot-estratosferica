from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import EditorialDecision


def save_queue(
    decisions: Iterable[EditorialDecision], output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "dry_run",
        "publishing_enabled": False,
        "items": [decision.to_dict() for decision in decisions],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
