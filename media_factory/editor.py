from __future__ import annotations

from typing import Any

from .editorial_quality import substantive_summary_issue
from .models import Candidate, EditorialDecision


def _hard_rejections(
    candidate: Candidate, config: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if not candidate.source_url:
        reasons.append("missing_source")
    if candidate.is_duplicate:
        reasons.append("duplicate_story")
    if not candidate.is_verified:
        reasons.append("unverified_rumor")
    if not candidate.has_media_rights:
        reasons.append("unlicensed_media_dependency")
    if not candidate.claims_supported:
        reasons.append("unsupported_claim")
    if candidate.territory not in config["territories"]:
        reasons.append("outside_editorial_territory")
    summary_issue = substantive_summary_issue(
        candidate.title,
        candidate.summary,
    )
    if summary_issue:
        reasons.append(summary_issue)
    configured = set(config.get("hard_reject", []))
    return [reason for reason in reasons if reason in configured]


def _score(candidate: Candidate, config: dict[str, Any]) -> dict[str, int]:
    weights = config["editorial_score"]["weights"]
    breakdown: dict[str, int] = {}
    for signal, weight in weights.items():
        raw_value = float(candidate.signals.get(signal, 0))
        normalized = min(1.0, max(0.0, raw_value))
        breakdown[signal] = round(normalized * int(weight))
    return breakdown


def evaluate_candidate(
    candidate: Candidate, config: dict[str, Any]
) -> EditorialDecision:
    rejection_reasons = _hard_rejections(candidate, config)
    breakdown = _score(candidate, config)
    score = sum(breakdown.values())
    minimum = int(config["editorial_score"]["minimum"])
    accepted = not rejection_reasons and score >= minimum
    state = "needs_review" if accepted else "draft"
    return EditorialDecision(
        title=candidate.title,
        score=score,
        state=state,
        accepted=accepted,
        rejection_reasons=rejection_reasons,
        score_breakdown=breakdown,
    )
