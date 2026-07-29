from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "brand_magnet_v1.json"
)
REQUIRED_OFFERS = {
    "brand_signal",
    "community_quest",
    "sponsored_multiverse",
}
REQUIRED_SAFETY = {
    "private_concepts_are_marked_unofficial": True,
    "sponsorship_is_always_disclosed": True,
    "automatic_outreach_enabled": False,
    "automatic_pricing_enabled": False,
    "human_approval_required": True,
    "publishing_enabled": False,
}


def load_brand_magnet(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_brand_magnet(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != "brand_magnet_v1":
        errors.append("invalid_brand_magnet_schema")

    offers = config.get("offers", [])
    offer_ids = {offer.get("id") for offer in offers}
    if offer_ids != REQUIRED_OFFERS:
        errors.append("incomplete_commercial_offers")
    for offer in offers:
        if not offer.get("outcome") or not offer.get("deliverables"):
            errors.append(f"incomplete_offer:{offer.get('id', 'unknown')}")

    qualification = config.get("qualification", {})
    if sum(qualification.get("weights", {}).values()) != 100:
        errors.append("qualification_weights_must_total_100")
    if not qualification.get("blocked_categories"):
        errors.append("missing_blocked_categories")

    safety = config.get("safety", {})
    for key, expected in REQUIRED_SAFETY.items():
        if safety.get(key) is not expected:
            errors.append(f"unsafe_brand_magnet_setting:{key}")
    return sorted(set(errors))


def qualify_brand_opportunity(
    opportunity: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = config or load_brand_magnet()
    errors = validate_brand_magnet(selected)
    if errors:
        raise ValueError(f"invalid brand magnet: {', '.join(errors)}")

    category = str(opportunity.get("category", "")).strip()
    if category in selected["qualification"]["blocked_categories"]:
        return {
            "status": "rejected",
            "score": 0,
            "offer_id": None,
            "reasons": ["blocked_category"],
            "automatic_outreach_enabled": False,
            "requires_human_review": True,
        }

    weights = selected["qualification"]["weights"]
    breakdown = {}
    for signal, weight in weights.items():
        value = min(1.0, max(0.0, float(opportunity.get(signal, 0))))
        breakdown[signal] = round(value * weight, 2)
    score = round(sum(breakdown.values()))

    objective = str(opportunity.get("objective", "")).casefold()
    if any(word in objective for word in ("particip", "encuesta", "investig")):
        offer_id = "community_quest"
    elif any(word in objective for word in ("serie", "awareness", "afinidad")):
        offer_id = "sponsored_multiverse"
    else:
        offer_id = "brand_signal"

    minimum = selected["qualification"]["minimum_score"]
    return {
        "status": "qualified" if score >= minimum else "research_only",
        "score": score,
        "score_breakdown": breakdown,
        "offer_id": offer_id,
        "reasons": [
            signal for signal, value in breakdown.items() if value > 0
        ],
        "automatic_outreach_enabled": False,
        "requires_human_review": True,
    }


def build_private_concept(
    *,
    brand: str,
    business_goal: str,
    gamer_tension: str,
    concept_name: str,
    mechanic: str,
    proof_plan: list[str],
) -> dict[str, Any]:
    """Create an internal concept record that cannot imply a partnership."""
    return {
        "schema_version": "private_brand_concept_v1",
        "brand": brand.strip(),
        "status": "private_unofficial_concept",
        "disclaimer": (
            "Concepto privado no oficial. No implica relación, autorización "
            "ni patrocinio de la marca."
        ),
        "business_goal": business_goal.strip(),
        "gamer_tension": gamer_tension.strip(),
        "concept_name": concept_name.strip(),
        "mechanic": mechanic.strip(),
        "proof_plan": [item.strip() for item in proof_plan if item.strip()],
        "publishing_enabled": False,
        "outreach_enabled": False,
        "requires_human_approval": True,
    }
