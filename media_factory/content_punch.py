from __future__ import annotations

import re
from typing import Any

from .models import Candidate


AUDIENCE_BY_TERRITORY = {
    "gaming_esports": "gamer y comunidad competitiva latinoamericana",
    "sport_technology_entertainment": (
        "aficionado que busca entender el espectáculo y sus datos"
    ),
    "ai_innovation_future": (
        "creador, profesional y decisor que necesita entender el impacto"
    ),
    "brands_activations": (
        "comunidad y gerente de mercadeo que evalúan utilidad y oportunidad"
    ),
}

PROMISE_BY_TERRITORY = {
    "gaming_esports": (
        "Explicar qué cambia para quien juega, compite o sigue la escena."
    ),
    "sport_technology_entertainment": (
        "Traducir la innovación en una consecuencia visible para el público."
    ),
    "ai_innovation_future": (
        "Convertir la novedad en una decisión útil para trabajo y negocio."
    ),
    "brands_activations": (
        "Separar una experiencia útil de una acción que solo hace ruido."
    ),
}

VISUAL_ENERGY_BY_TERRITORY = {
    "gaming_esports": "cinética, competitiva y de alto contraste",
    "sport_technology_entertainment": "acción, dato protagonista y tensión",
    "ai_innovation_future": "tecnológica, humana y basada en evidencia",
    "brands_activations": "editorial, aspiracional y orientada a resultados",
}

AI_WORK_TERMS = {
    "trabajo",
    "empleo",
    "productividad",
    "automatización",
    "colaboración",
    "interacciones",
}


def _is_high_impact(candidate: Candidate) -> bool:
    signals = candidate.signals
    return (
        float(signals.get("conversation_potential", 0)) >= 0.8
        and float(signals.get("angle_originality", 0)) >= 0.75
    )


def _story_text(candidate: Candidate) -> str:
    return f"{candidate.title} {candidate.summary}".casefold()


def _contextual_hook(candidate: Candidate, high_impact: bool) -> str:
    text = _story_text(candidate)
    if (
        candidate.territory == "ai_innovation_future"
        and "ia" in text
        and any(term in text for term in AI_WORK_TERMS)
    ):
        return "¿LA IA TE POTENCIA O TE REEMPLAZA?"
    if high_impact:
        return f"ESTO CAMBIA LA CONVERSACIÓN: {candidate.title}"
    return candidate.title


def _verified_numeric_value(candidate: Candidate) -> str | None:
    """Lift numeric facts from verified copy without generating new claims."""
    source = candidate.summary.strip() or candidate.title.strip()
    patterns = (
        r"\b\d+(?:[.,]\d+)?\s+millones?\s+de\s+[\wáéíóúüñ-]+",
        r"\bmás\s+de\s+\d+(?:[.,]\d+)?\s+[\wáéíóúüñ-]+",
    )
    facts: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            fact = match.group(0).strip(" ,.;:")
            if fact.casefold() not in {item.casefold() for item in facts}:
                facts.append(fact)
    if not facts:
        return None
    return " · ".join(fact.upper() for fact in facts[:2])


def build_content_punch(
    candidate: Candidate,
    audience_experiment: dict[str, Any],
) -> dict[str, Any]:
    """Build a grounded impact plan; it never invents supporting evidence."""
    high_impact = _is_high_impact(candidate)
    question = str(audience_experiment["learning_question"]).strip()
    options = [
        str(option).strip()
        for option in audience_experiment["answer_options"]
        if str(option).strip()
    ]
    evidence = candidate.summary.strip() or candidate.title.strip()
    hook = _contextual_hook(candidate, high_impact)
    concrete_value = _verified_numeric_value(candidate) or evidence
    expected_action = (
        f"Responder: {' / '.join(options)}"
        if options
        else "Responder la pregunta con una experiencia concreta."
    )

    plan = {
        "primary_audience": AUDIENCE_BY_TERRITORY[candidate.territory],
        "hook": hook,
        "concrete_value": concrete_value,
        "evidence_origin": (
            "candidate.summary" if candidate.summary.strip() else "candidate.title"
        ),
        "audience_promise": PROMISE_BY_TERRITORY[candidate.territory],
        "tension_question": question,
        "expected_action": expected_action,
        "tone": "high_impact" if high_impact else "analytical",
        "visual_energy": VISUAL_ENERGY_BY_TERRITORY[candidate.territory],
        "quality_requirements": [
            "hook",
            "concrete_value",
            "tension_question",
            "expected_action",
        ],
        "requires_human_review": True,
        "publishing_enabled": False,
    }
    plan["gate_passed"] = not validate_content_punch(plan)
    return plan


def validate_content_punch(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "primary_audience",
        "hook",
        "concrete_value",
        "evidence_origin",
        "audience_promise",
        "tension_question",
        "expected_action",
        "tone",
        "visual_energy",
    ):
        if not str(plan.get(field, "")).strip():
            errors.append(f"missing_punch_field:{field}")
    if plan.get("evidence_origin") not in {
        "candidate.summary",
        "candidate.title",
    }:
        errors.append("unsupported_evidence_origin")
    if plan.get("tone") not in {"analytical", "high_impact"}:
        errors.append("invalid_punch_tone")
    if not plan.get("requires_human_review"):
        errors.append("punch_human_review_required")
    if plan.get("publishing_enabled") is not False:
        errors.append("punch_publishing_must_be_disabled")
    return errors
