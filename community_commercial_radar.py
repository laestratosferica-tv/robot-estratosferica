from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


COMMERCIAL_SIGNALS = {
    "cotización": 24,
    "cotizacion": 24,
    "propuesta": 20,
    "presupuesto": 24,
    "pauta": 22,
    "patrocinio": 24,
    "patrocinar": 24,
    "campaña": 18,
    "campana": 18,
    "evento": 15,
    "transmisión": 18,
    "transmision": 18,
    "colaboración": 16,
    "colaboracion": 16,
    "alianza": 18,
    "branded content": 22,
    "contenido de marca": 22,
    "activación": 20,
    "activacion": 20,
    "quiero contratar": 30,
    "nos interesa": 24,
    "precio": 14,
    "tarifa": 18,
    "contacto comercial": 22,
    "media kit": 22,
}

BUYING_SIGNALS = {
    "cómo compramos": 24,
    "como compramos": 24,
    "cómo contrato": 26,
    "como contrato": 26,
    "dónde reservo": 20,
    "donde reservo": 20,
    "quiero una demo": 20,
    "agenda": 12,
    "reunión": 15,
    "reunion": 15,
    "escríbeme": 12,
    "escribeme": 12,
}

BUSINESS_IDENTIFIERS = {
    "mi marca": 12,
    "nuestra marca": 14,
    "mi empresa": 12,
    "nuestra empresa": 14,
    "mi agencia": 12,
    "somos una marca": 14,
    "represento a": 12,
}

RISK_SIGNALS = {
    "dinero fácil",
    "ingreso garantizado",
    "apuesta segura",
    "casino",
    "crypto pump",
    "sígueme y te sigo",
    "follow for follow",
    "envíame un código",
    "enviame un codigo",
}

INTENT_RULES = [
    ("patrocinio", {"patrocinio", "patrocinar", "sponsor"}),
    ("pauta", {"pauta", "media kit", "tarifa", "publicidad"}),
    ("branded_content", {"branded content", "contenido de marca", "campaña", "campana"}),
    ("eventos", {"evento", "activación", "activacion", "torneo"}),
    ("transmisiones", {"transmisión", "transmision", "streaming", "en vivo"}),
    ("servicios_creativos", {"producción", "produccion", "creatividad", "agencia", "video"}),
    ("alianzas", {"alianza", "colaboración", "colaboracion"}),
    ("compra", {"precio", "comprar", "reservar", "contratar", "demo"}),
]


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _text(interaction: Dict[str, Any]) -> str:
    return _normalize(" ".join([
        str(interaction.get("text", "")),
        str(interaction.get("bio", "")),
        str(interaction.get("display_name", "")),
    ]))


def _matches(text: str, weighted_terms: Dict[str, int]) -> List[tuple[str, int]]:
    return [(term, weight) for term, weight in weighted_terms.items() if term in text]


def _intent(text: str) -> str:
    for intent, terms in INTENT_RULES:
        if any(term in text for term in terms):
            return intent
    return "conversación"


def _reply_draft(display_name: str, intent: str) -> str:
    greeting = f"Hola, {display_name}. " if display_name else "Hola. "
    if intent == "conversación":
        return greeting + "Gracias por sumarte a la conversación. ¿Qué parte te interesa explorar más?"
    return (
        greeting
        + "Gracias por pensar en La Estratosférica. Para entender bien la oportunidad, "
        + "¿nos cuentas brevemente la marca, el objetivo y el momento estimado del proyecto?"
    )


def classify_interaction(
    interaction: Dict[str, Any],
    *,
    minimum_commercial_score: int = 25,
    hot_score: int = 60,
) -> Dict[str, Any]:
    """Classify one public interaction without performing any outbound action."""
    text = _text(interaction)
    risk_hits = sorted(term for term in RISK_SIGNALS if term in text)

    if risk_hits:
        return {
            **interaction,
            "classification": "risk_or_spam",
            "intent": "risk",
            "score": 0,
            "temperature": "none",
            "reasons": risk_hits,
            "suggested_next_action": "ignore_or_review",
            "reply_draft": "",
            "requires_human_approval": True,
            "can_auto_send": False,
        }

    matches = (
        _matches(text, COMMERCIAL_SIGNALS)
        + _matches(text, BUYING_SIGNALS)
        + _matches(text, BUSINESS_IDENTIFIERS)
    )
    unique = {term: weight for term, weight in matches}
    score = min(100, sum(unique.values()))

    has_contact = bool(re.search(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", text))
    if has_contact:
        score = min(100, score + 15)
        unique["contacto_visible"] = 15

    commercial = score >= minimum_commercial_score
    intent = _intent(text)
    temperature = "hot" if score >= hot_score else ("warm" if commercial else "cold")

    return {
        **interaction,
        "classification": "commercial_lead" if commercial else "community_signal",
        "intent": intent,
        "score": score,
        "temperature": temperature,
        "reasons": sorted(unique),
        "suggested_next_action": (
            "jose_luis_review_for_contact" if commercial
            else "community_reply_review"
        ),
        "reply_draft": _reply_draft(
            str(interaction.get("display_name", "")).strip(), intent
        ),
        "requires_human_approval": True,
        "can_auto_send": False,
    }


def prioritize_interactions(
    interactions: Iterable[Dict[str, Any]],
    *,
    minimum_commercial_score: int = 25,
    hot_score: int = 60,
) -> List[Dict[str, Any]]:
    classified = [
        classify_interaction(
            item,
            minimum_commercial_score=minimum_commercial_score,
            hot_score=hot_score,
        )
        for item in interactions
    ]
    class_priority = {"commercial_lead": 0, "community_signal": 1, "risk_or_spam": 2}
    return sorted(
        classified,
        key=lambda item: (
            class_priority[item["classification"]],
            -int(item["score"]),
            str(item.get("created_at", "")),
        ),
    )


def prepare_outbound_action(
    lead: Dict[str, Any],
    *,
    human_approved: bool = False,
    allow_outbound: bool = False,
) -> Dict[str, Any]:
    """Create a safe handoff record. This module never sends a message."""
    ready = bool(
        human_approved
        and allow_outbound
        and lead.get("classification") != "risk_or_spam"
        and lead.get("reply_draft")
    )
    return {
        "status": "ready_for_manual_send" if ready else "blocked_pending_approval",
        "text": lead.get("reply_draft", "") if ready else "",
        "platform": lead.get("platform", ""),
        "interaction_id": lead.get("interaction_id", ""),
        "sent": False,
    }
