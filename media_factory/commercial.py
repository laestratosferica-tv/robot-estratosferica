from __future__ import annotations

from typing import Any

from .models import Candidate, CommercialOpportunity, EditorialDecision


OPPORTUNITY_KIND = {
    "gaming_esports": "gaming_partnership",
    "sport_technology_entertainment": "sportstech_activation",
    "ai_innovation_future": "innovation_brief",
    "brands_activations": "digital_activation",
}


def _signal(candidate: Candidate, name: str) -> float:
    return min(1.0, max(0.0, float(candidate.signals.get(name, 0))))


def detect_opportunity(
    candidate: Candidate,
    decision: EditorialDecision,
    minimum_score: int = 55,
) -> CommercialOpportunity | None:
    """Detecta una pista comercial; nunca contacta ni recomienda contactar."""
    if not decision.accepted:
        return None
    score = round(
        _signal(candidate, "commercial_potential") * 50
        + _signal(candidate, "latam_relevance") * 20
        + _signal(candidate, "explanatory_value") * 15
        + _signal(candidate, "conversation_potential") * 15
    )
    if score < minimum_score:
        return None
    kind = OPPORTUNITY_KIND.get(candidate.territory, "editorial_insight")
    return CommercialOpportunity(
        kind=kind,
        score=score,
        status="research_only",
        rationale=(
            "La historia combina relevancia regional, valor editorial "
            "y potencial comercial medible."
        ),
        next_step=(
            "Documentar marcas, comunidades y necesidades relacionadas; "
            "no enviar mensajes sin aprobación humana."
        ),
    )
