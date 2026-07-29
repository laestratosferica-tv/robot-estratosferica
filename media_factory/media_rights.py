from __future__ import annotations

from typing import Any

from .models import Candidate


def qualifies_for_editorial_citation(
    candidate: Candidate,
    config: dict[str, Any],
) -> bool:
    """Validate the documented, human-reviewed editorial citation pathway."""
    policy = config.get("editorial_citation_policy", {})
    usage = candidate.media_usage
    if not policy.get("enabled") or usage.get("pathway") != "editorial_citation":
        return False
    allowed_purposes = set(policy.get("allowed_purposes", []))
    if usage.get("purpose") not in allowed_purposes:
        return False
    required_flags = (
        "minimum_necessary",
        "transformative_commentary",
        "source_attribution",
        "source_link_preserved",
        "original_music_removed",
        "uploader_identity_verified",
        "human_review_required",
    )
    if not all(usage.get(flag) is True for flag in required_flags):
        return False

    excerpt_seconds = usage.get("max_excerpt_seconds")
    internal_limit = policy.get("internal_max_excerpt_seconds")
    if not isinstance(excerpt_seconds, (int, float)) or excerpt_seconds <= 0:
        return False
    if not isinstance(internal_limit, (int, float)):
        return False
    return excerpt_seconds <= internal_limit


def has_usable_media_path(
    candidate: Candidate,
    config: dict[str, Any],
) -> bool:
    return candidate.has_media_rights or qualifies_for_editorial_citation(
        candidate,
        config,
    )
