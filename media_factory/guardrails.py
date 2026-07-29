from __future__ import annotations

import re
from urllib.parse import urlparse

from .audience_intelligence import story_question
from .editorial_quality import (
    normalize_editorial_text,
    substantive_summary_issue,
    text_is_equivalent,
    unsupported_context_domains,
)
from .models import Candidate, ContentPackage, Storyboard
from .content_punch import validate_content_punch


PLATFORM_LIMITS = {
    "instagram": 2200,
    "facebook": 3000,
    "youtube": 5000,
    "threads": 500,
}
SHORT_VIDEO_WORD_LIMIT = 75


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
    if text_is_equivalent(package.headline, package.factual_summary):
        errors.append("factual_summary_repeats_headline")
    if text_is_equivalent(
        package.headline,
        str(package.content_punch.get("concrete_value", "")),
    ):
        errors.append("concrete_value_repeats_headline")
    if len(re.findall(r"\b[\wáéíóúüñ]+\b", package.short_video_script)) > (
        SHORT_VIDEO_WORD_LIMIT
    ):
        errors.append("short_video_script_exceeds_word_budget")
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


SAFE_STORYBOARD_LABELS = {
    "Hecho confirmado",
    "Lectura editorial",
    "Consecuencias por evaluar",
    "Sin completar vacíos",
}

SAFE_EDITORIAL_TRANSITIONS = {
    (
        "Lectura editorial: esta pieza separa el hecho confirmado de sus "
        "posibles consecuencias."
    ),
    (
        "Lectura editorial: conviene distinguir el hecho confirmado de las "
        "consecuencias que todavía deben evaluarse."
    ),
    (
        "Lectura editorial: cualquier consecuencia debe comprobarse a partir "
        "de la evidencia disponible."
    ),
}


def _is_extract_supported(text: str, evidence: str) -> bool:
    generated_tokens = set(normalize_editorial_text(text).split())
    evidence_tokens = set(normalize_editorial_text(evidence).split())
    return bool(generated_tokens) and generated_tokens <= evidence_tokens


def validate_evidence_alignment(
    candidate: Candidate,
    package: ContentPackage,
    storyboard: Storyboard,
) -> list[str]:
    """Final deterministic gate for factual grounding and story congruence."""
    errors: list[str] = []
    summary_issue = substantive_summary_issue(
        candidate.title,
        candidate.summary,
    )
    if summary_issue:
        errors.append(summary_issue)
    if package.factual_summary != candidate.summary:
        errors.append("package_summary_not_candidate_evidence")
    if package.headline != candidate.title:
        errors.append("headline_not_extractively_grounded")

    expected_question, expected_options = story_question(candidate)
    experiment = package.audience_experiment
    actual_question = str(experiment.get("learning_question", "")).strip()
    actual_options = [
        str(option).strip()
        for option in experiment.get("answer_options", [])
    ]
    if actual_question != expected_question:
        errors.append("question_incongruent_with_story_type")
    if actual_options != expected_options:
        errors.append("question_options_incongruent_with_story_type")
    if package.content_punch.get("tension_question") != expected_question:
        errors.append("content_question_not_grounded")

    concrete_value = str(
        package.content_punch.get("concrete_value", "")
    ).strip()
    if not _is_extract_supported(concrete_value, candidate.summary):
        errors.append("concrete_value_not_extractively_grounded")
    short_video_context = str(
        package.content_punch.get("short_video_context", "")
    ).strip()
    if not _is_extract_supported(short_video_context, candidate.summary):
        errors.append("short_video_context_not_extractively_grounded")

    generated_copy = " ".join([
        package.headline,
        package.angle,
        package.factual_summary,
        package.short_video_script,
        *package.platform_copy.values(),
        *package.visual_brief,
    ])
    evidence = f"{candidate.title} {candidate.summary}"
    for domain in unsupported_context_domains(evidence, generated_copy):
        errors.append(f"unsupported_generated_context:{domain}")

    allowed_voiceovers = {
        candidate.title,
        candidate.summary,
        short_video_context,
        expected_question,
        *SAFE_EDITORIAL_TRANSITIONS,
    }
    allowed_screen_text = {
        candidate.title,
        expected_question,
        *SAFE_STORYBOARD_LABELS,
    }
    for scene in storyboard.scenes:
        if scene.voiceover not in allowed_voiceovers:
            errors.append(
                f"unsupported_storyboard_voiceover:{scene.scene_id}"
            )
        if scene.on_screen_text not in allowed_screen_text:
            errors.append(
                f"unsupported_storyboard_text:{scene.scene_id}"
            )

    storyboard_text = " ".join(
        value
        for scene in storyboard.scenes
        for value in (
            scene.voiceover,
            scene.on_screen_text,
            scene.visual_direction,
        )
    )
    for domain in unsupported_context_domains(evidence, storyboard_text):
        errors.append(f"unsupported_storyboard_context:{domain}")
    return sorted(set(errors))
