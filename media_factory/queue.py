from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import PipelineItem
from .strategy import validate_strategy_decision


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
    strategic_classification = dict(
        item.candidate.strategic_classification
    )
    strategy_errors = validate_strategy_decision(
        strategic_classification
    )
    if strategy_errors:
        raise ValueError(
            "Review item blocked by strategy gate: "
            + ", ".join(strategy_errors)
        )
    selection = item.opportunity_selection
    if selection is None or not selection.selected:
        raise ValueError(
            "Review item blocked: opportunity was not selected"
        )
    if not selection.eligible or selection.views_only_success_allowed:
        raise ValueError(
            "Review item blocked by opportunity selector gate"
        )
    if selection.publishing_enabled or selection.external_actions_enabled:
        raise ValueError(
            "Review item blocked: selector attempted an external action"
        )
    experiment = (
        package.audience_experiment
        if package
        else {}
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
        "strategy": strategic_classification,
        "opportunity_selection": selection.to_dict(),
        "editorial_test": {
            "state": "draft",
            "objective": selection.objective,
            "expected_interaction": selection.expected_interaction,
            "interaction_prompt": experiment.get(
                "learning_question", ""
            ),
            "answer_options": experiment.get("answer_options", []),
            "primary_metric": selection.primary_metric,
            "audience_hypothesis": selection.audience_hypothesis,
            "views_only_success_allowed": False,
            "requires_human_approval": True,
            "publishing_enabled": False,
            "external_actions_enabled": False,
        },
    }
    return payload


def save_queue(
    items: Iterable[PipelineItem],
    output_path: str | Path,
    *,
    selection_report: dict | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "review_queue_v1",
        "mode": "dry_run",
        "publishing_enabled": False,
        "external_actions_enabled": False,
        "human_approval_required": True,
        "opportunity_selection": selection_report or {},
        "items": [_review_record(item) for item in items],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
