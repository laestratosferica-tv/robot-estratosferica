from __future__ import annotations

from urllib.parse import urlparse

from .models import ContentPackage


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
    return errors
