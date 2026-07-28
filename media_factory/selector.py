from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .models import Candidate, EditorialDecision, OpportunitySelection
from .strategy import validate_strategy_decision


class OpportunitySelectorError(ValueError):
    pass


SIGNAL_LABELS = {
    "editorial_quality": "calidad editorial",
    "conversation_potential": "potencial de conversación",
    "explanatory_value": "valor explicativo",
    "commercial_potential": "potencial comercial",
    "latam_relevance": "relevancia LATAM",
    "angle_originality": "originalidad del ángulo",
}


def _candidate_id(candidate: Candidate) -> str:
    if candidate.candidate_id:
        return candidate.candidate_id
    digest = hashlib.sha256(
        f"{candidate.source_id}\0{candidate.source_url}".encode("utf-8")
    ).hexdigest()
    return f"candidate-{digest[:20]}"


def _signal(candidate: Candidate, name: str) -> float:
    raw = float(candidate.signals.get(name, 0))
    return min(1.0, max(0.0, raw))


def _selector_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    selector = config.get("opportunity_selector", {})
    weights = selector.get("weights", {})
    required = set(SIGNAL_LABELS)
    if set(weights) != required:
        raise OpportunitySelectorError(
            "El selector debe configurar exactamente sus seis dimensiones"
        )
    if sum(float(value) for value in weights.values()) != 100:
        raise OpportunitySelectorError(
            "Los pesos del selector deben sumar 100"
        )
    minimum = selector.get("minimum_score")
    if not isinstance(minimum, (int, float)) or not 0 <= minimum <= 100:
        raise OpportunitySelectorError(
            "El mínimo del selector debe estar entre 0 y 100"
        )
    if selector.get("max_selected_per_run") != 1:
        raise OpportunitySelectorError(
            "La prueba controlada solo puede seleccionar una oportunidad"
        )
    if selector.get("publishing_enabled") is not False:
        raise OpportunitySelectorError(
            "El selector debe mantener publicación desactivada"
        )
    if selector.get("external_actions_enabled") is not False:
        raise OpportunitySelectorError(
            "El selector debe mantener acciones externas desactivadas"
        )
    if selector.get("human_approval_required") is not True:
        raise OpportunitySelectorError(
            "El selector debe exigir aprobación humana"
        )
    if selector.get("views_only_success_allowed") is not False:
        raise OpportunitySelectorError(
            "El selector no puede considerar las vistas como éxito único"
        )
    return selector


def _blocking_reasons(
    candidate: Candidate,
    decision: EditorialDecision,
) -> list[str]:
    reasons = list(decision.rejection_reasons)
    if not decision.accepted and not reasons:
        reasons.append("editorial_score_below_minimum")
    strategy = candidate.strategic_classification
    reasons.extend(validate_strategy_decision(strategy))
    if strategy.get("rights_ready_for_draft") is not True:
        reasons.append("rights_not_ready_for_draft")
    if strategy.get("primary_metric") in {"views", "reproducciones"}:
        reasons.append("views_only_metric_not_allowed")
    return sorted(set(reasons))


def _score_breakdown(
    candidate: Candidate,
    decision: EditorialDecision,
    selector: Mapping[str, Any],
) -> dict[str, float]:
    weights = selector["weights"]
    normalized = {
        "editorial_quality": min(1.0, max(0.0, decision.score / 100)),
        "conversation_potential": _signal(
            candidate, "conversation_potential"
        ),
        "explanatory_value": _signal(candidate, "explanatory_value"),
        "commercial_potential": _signal(candidate, "commercial_potential"),
        "latam_relevance": _signal(candidate, "latam_relevance"),
        "angle_originality": _signal(candidate, "angle_originality"),
    }
    return {
        name: round(normalized[name] * float(weights[name]), 3)
        for name in SIGNAL_LABELS
    }


def _rationale(breakdown: Mapping[str, float]) -> list[str]:
    strongest = sorted(
        breakdown,
        key=lambda name: (-breakdown[name], name),
    )[:3]
    return [
        f"{SIGNAL_LABELS[name]}:{breakdown[name]:g}"
        for name in strongest
        if breakdown[name] > 0
    ]


def rank_opportunities(
    candidates: Sequence[Candidate],
    decisions: Sequence[EditorialDecision],
    config: Mapping[str, Any],
) -> list[OpportunitySelection]:
    if len(candidates) != len(decisions):
        raise OpportunitySelectorError(
            "Cada candidato debe tener una decisión editorial"
        )
    selector = _selector_config(config)
    minimum = float(selector["minimum_score"])
    provisional: list[OpportunitySelection] = []

    for candidate, decision in zip(candidates, decisions):
        strategy = candidate.strategic_classification
        blocking = _blocking_reasons(candidate, decision)
        breakdown = _score_breakdown(candidate, decision, selector)
        score = round(sum(breakdown.values()), 3)
        eligible = decision.accepted and not blocking and score >= minimum
        if decision.accepted and not blocking and score < minimum:
            blocking.append("below_selector_minimum")
        candidate_id = _candidate_id(candidate)
        provisional.append(
            OpportunitySelection(
                selection_id=f"selection-{candidate_id}",
                candidate_id=candidate_id,
                candidate_title=candidate.title,
                content_product_id=str(
                    strategy.get("content_product_id", "")
                ),
                rank=None,
                score=score,
                status="eligible" if eligible else "ineligible",
                selected=False,
                eligible=eligible,
                score_breakdown=breakdown,
                rationale=_rationale(breakdown),
                blocking_reasons=blocking,
                objective=str(strategy.get("purpose", "")),
                expected_interaction=str(
                    strategy.get("expected_community_action", "")
                ),
                primary_metric=str(strategy.get("primary_metric", "")),
                audience_hypothesis=str(
                    strategy.get("audience_hypothesis", "")
                ),
            )
        )

    ranked_indexes = sorted(
        range(len(provisional)),
        key=lambda index: (
            not provisional[index].eligible,
            -provisional[index].score,
            -decisions[index].score,
            provisional[index].candidate_id,
        ),
    )
    rank = 0
    selected = False
    result = list(provisional)
    for index in ranked_indexes:
        item = result[index]
        if not item.eligible:
            continue
        rank += 1
        is_selected = not selected
        selected = selected or is_selected
        result[index] = replace(
            item,
            rank=rank,
            selected=is_selected,
            status="selected" if is_selected else "eligible_not_selected",
        )
    return result


def build_selection_report(
    selections: Iterable[OpportunitySelection],
) -> dict[str, Any]:
    items = list(selections)
    ranked = sorted(
        items,
        key=lambda item: (
            item.rank is None,
            item.rank or 10**9,
            -item.score,
            item.candidate_id,
        ),
    )
    selected = [item for item in ranked if item.selected]
    if len(selected) > 1:
        raise OpportunitySelectorError(
            "La prueba controlada no puede seleccionar más de una oportunidad"
        )
    return {
        "schema_version": "opportunity_selection_v1",
        "mode": "deterministic_zero_cost",
        "candidate_count": len(items),
        "eligible_count": sum(item.eligible for item in items),
        "selected_count": len(selected),
        "selected_candidate_id": (
            selected[0].candidate_id if selected else None
        ),
        "views_only_success_allowed": False,
        "publishing_enabled": False,
        "external_actions_enabled": False,
        "human_approval_required": True,
        "ranked_candidates": [item.to_dict() for item in ranked],
    }
