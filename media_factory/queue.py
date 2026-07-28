from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import PipelineItem


def _stable_id(*parts: str) -> str:
    normalized = "\0".join(part.strip() for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _review_record(item: PipelineItem) -> dict:
    payload = item.to_dict()
    package = item.content_package
    platform_copy = dict(package.platform_copy) if package else {}
    candidate_id = item.candidate.candidate_id or _stable_id(
        item.candidate.source_id,
        item.candidate.source_url,
    )
    fingerprint = _stable_id(
        item.candidate.source_url,
        json.dumps(platform_copy, ensure_ascii=False, sort_keys=True),
    )
    payload["review"] = {
        "review_id": f"review-{candidate_id[:16]}",
        "candidate_id": candidate_id,
        "content_fingerprint": fingerprint,
        "anti_duplicate_id": f"content-{fingerprint[:20]}",
        "status": "pending_human_approval",
        "requires_human_approval": True,
        "approved": False,
        "publish_allowed": False,
        "source": {
            "name": item.candidate.source_id,
            "url": item.candidate.source_url,
            "published_at": item.candidate.published_at,
        },
        "final_text_by_platform": platform_copy,
    }
    return payload


def save_queue(
    items: Iterable[PipelineItem], output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "review_queue_v1",
        "mode": "dry_run",
        "publishing_enabled": False,
        "external_actions_enabled": False,
        "human_approval_required": True,
        "items": [_review_record(item) for item in items],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
