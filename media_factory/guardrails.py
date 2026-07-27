from __future__ import annotations

from urllib.parse import urlparse

from .models import ContentPackage, Storyboard
from .content_punch import validate_content_punch


PLATFORM_LIMITS = {
    "instagram": 2200,
    "facebook": 3000,
    "youtube": 5000,
    "threads": 500,
}


def validate_content_package(package: ContentPackage) -> list[str]:
    errors: list[str] = []
    if package.state != "draft":
        errors.append("package_must_remain_draft")
    if not package.requires_human_review:
        errors.append("human_review_required")
    if package.external_actions_enabled:
        errors.append("external_actions_must_be_disabled")
    if set(package.platform_copy) != set(PLATFORM_LIMITS):
        errors.append("unexpected_platform_set")
    for platform, text in package.platform_copy.items():
        limit = PLATFORM_LIMITS.get(platform, 0)
        if not text or len(text) > limit:
            errors.append(f"invalid_copy_length:{platform}")
    if not package.sources:
        errors.append("missing_sources")
    for source in package.sources:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append("invalid_source_url")
    visual_text = " ".join(package.visual_brief).lower()
    if "original" not in visual_text or "no descargar" not in visual_text:
        errors.append("missing_visual_rights_instruction")
    punch_errors = validate_content_punch(package.content_punch)
    errors.extend(punch_errors)
    if not package.content_punch.get("gate_passed"):
        errors.append("content_punch_gate_failed")
    for required_label in (
        "gancho dominante:",
        "valor concreto visible:",
        "pregunta o tensión visible:",
        "acción esperada:",
    ):
        if required_label not in visual_text:
            errors.append(f"missing_visual_punch_instruction:{required_label}")
    return errors


def validate_storyboard(storyboard: Storyboard) -> list[str]:
    errors: list[str] = []
    if storyboard.state != "draft":
        errors.append("storyboard_must_remain_draft")
    if storyboard.production_enabled:
        errors.append("production_must_be_disabled")
    if not storyboard.requires_human_review:
        errors.append("storyboard_human_review_required")
    if storyboard.master_format != "1080x1920":
        errors.append("invalid_master_format")
    if storyboard.duration_seconds > 30:
        errors.append("duration_exceeds_pilot_limit")
    if not storyboard.captions_required:
        errors.append("captions_required")
    if not storyboard.scenes:
        errors.append("missing_scenes")
    else:
        if storyboard.scenes[0].start_second != 0:
            errors.append("storyboard_must_start_at_zero")
        if storyboard.scenes[-1].end_second != storyboard.duration_seconds:
            errors.append("storyboard_duration_mismatch")
        for previous, current in zip(
            storyboard.scenes, storyboard.scenes[1:]
        ):
            if previous.end_second != current.start_second:
                errors.append("storyboard_timeline_gap")
    if not storyboard.source_card.get("url"):
        errors.append("missing_storyboard_source")
    style_text = " ".join(storyboard.visual_style).lower()
    if "original" not in style_text or "sin logos" not in style_text:
        errors.append("missing_original_visual_policy")
    return errors
