from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import Candidate


class StrategyConfigurationError(ValueError):
    pass


REQUIRED_DECISION_FIELDS = {
    "content_product_id",
    "purpose",
    "audience_hypothesis",
    "expected_community_action",
    "primary_metric",
    "commercial_path",
    "rights_state",
}

PRODUCT_AUDIENCE_HYPOTHESES = {
    "radar_estratosferico": (
        "La audiencia gamer participa cuando entiende rápido por qué una "
        "noticia cambia su forma de jugar, elegir o conversar."
    ),
    "jugada_estratosferica": (
        "Una jugada excepcional provoca retención y debate cuando conserva "
        "su contexto, autoría y estado de derechos."
    ),
    "comunidad_decide": (
        "Las opciones concretas convierten una opinión dispersa en una señal "
        "de afinidad que puede medirse."
    ),
    "esto_cambia_el_juego": (
        "La audiencia guarda y comparte un cambio tecnológico cuando se "
        "explica su consecuencia práctica para gamers latinoamericanos."
    ),
    "cultura_en_modo_gamer": (
        "Moda, gastronomía o entretenimiento generan afinidad cuando su "
        "conexión con la vida gamer es explícita y útil."
    ),
    "setup_real": (
        "Una comparación transparente de tecnología revela intención de "
        "compra y ayuda a tomar una decisión real."
    ),
    "arena_estratosferica": (
        "Una transmisión autorizada crea hábito cuando invita a ver, "
        "participar, registrarse y regresar."
    ),
    "torneo_estratosferico": (
        "La competencia propia convierte espectadores en participantes y "
        "construye comunidad y derechos propios."
    ),
    "informe_de_audiencia": (
        "Las métricas agregadas permiten convertir aprendizaje editorial en "
        "argumentos comerciales verificables."
    ),
}

SETUP_TERMS = {
    "mouse",
    "ratón gamer",
    "teclado",
    "monitor",
    "portátil",
    "laptop",
    "computador",
    "pc gamer",
    "gpu",
    "tarjeta gráfica",
    "audífonos",
    "headset",
    "silla gamer",
    "periférico",
}
BUYING_TERMS = {
    "comparativa",
    "review",
    "reseña",
    "precio",
    "comprar",
    "vale la pena",
    "mejor",
    "prueba",
}
TECH_CHANGE_TERMS = {
    "inteligencia artificial",
    " ia ",
    "gemini",
    "tecnología",
    "technology",
    "innovación",
    "developer",
    "desarrollador",
    "cloud gaming",
    "compatibilidad",
    "ecosistema",
}
TECH_IMPACT_TERMS = {
    "cambia",
    "impacto",
    "futuro",
    "trabajo",
    "privacidad",
    "seguridad",
    "acceso",
    "herramienta",
    "plataforma",
}
LIFESTYLE_TERMS = {
    "moda",
    "ropa",
    "tenis",
    "sneakers",
    "café",
    "coffee",
    "comida",
    "gastronomía",
    "restaurante",
    "música",
    "cine",
    "serie",
    "entretenimiento",
}
GAMER_CONNECTION_TERMS = {
    "gamer",
    "gaming",
    "videojuego",
    "esports",
    "streamer",
    "twitch",
    "xbox",
    "playstation",
    "nintendo",
    "valorant",
    "league of legends",
    "fortnite",
}


def load_content_strategy(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        strategy = json.load(handle)
    validate_content_strategy(strategy)
    return strategy


def validate_content_strategy(strategy: Mapping[str, Any]) -> None:
    products = strategy.get("content_products")
    if not isinstance(products, list) or not products:
        raise StrategyConfigurationError(
            "La matriz debe incluir productos editoriales"
        )
    product_ids = {str(product.get("id", "")) for product in products}
    required_products = {
        "radar_estratosferico",
        "jugada_estratosferica",
        "comunidad_decide",
        "esto_cambia_el_juego",
        "cultura_en_modo_gamer",
        "setup_real",
        "arena_estratosferica",
        "torneo_estratosferico",
        "informe_de_audiencia",
    }
    if not required_products <= product_ids:
        raise StrategyConfigurationError(
            "La matriz no contiene todos los productos estratégicos"
        )
    configured_fields = set(strategy.get("required_decision_fields", []))
    if not REQUIRED_DECISION_FIELDS <= configured_fields:
        raise StrategyConfigurationError(
            "La matriz no exige todos los campos de decisión"
        )
    safety = strategy.get("safety", {})
    if safety.get("publishing_enabled") is not False:
        raise StrategyConfigurationError(
            "La clasificación exige publishing_enabled=false"
        )
    if safety.get("broadcasting_enabled") is not False:
        raise StrategyConfigurationError(
            "La clasificación exige broadcasting_enabled=false"
        )
    if safety.get("automatic_commercial_outreach_enabled") is not False:
        raise StrategyConfigurationError(
            "La clasificación exige contacto comercial automático apagado"
        )
    if safety.get("human_approval_required") is not True:
        raise StrategyConfigurationError(
            "La clasificación exige aprobación humana"
        )


def validate_strategy_decision(decision: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing_strategy_field:{field}"
        for field in sorted(REQUIRED_DECISION_FIELDS)
        if not decision.get(field)
    ]
    if decision.get("requires_human_review") is not True:
        errors.append("strategy_must_require_human_review")
    if decision.get("external_actions_enabled") is not False:
        errors.append("strategy_external_actions_must_remain_disabled")
    if decision.get("publishing_enabled") is not False:
        errors.append("strategy_publishing_must_remain_disabled")
    if decision.get("broadcasting_enabled") is not False:
        errors.append("strategy_broadcasting_must_remain_disabled")
    return errors


def _candidate_data(candidate: Candidate | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, Candidate):
        return {
            "title": candidate.title,
            "summary": candidate.summary,
            "territory": candidate.territory,
            "source_id": candidate.source_id,
            "source_url": candidate.source_url,
            "has_media_rights": candidate.has_media_rights,
        }
    return dict(candidate)


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _select_product(data: Mapping[str, Any]) -> tuple[str, list[str]]:
    text = (
        f" {data.get('title', '')} "
        f"{data.get('summary', data.get('description', ''))} "
    ).casefold()
    lane = str(data.get("editorial_lane", "")).casefold()
    content_type = str(data.get("content_type", "")).casefold()
    matched: list[str] = []

    if lane == "epic_plays_and_creators" or content_type == "epic_play":
        return "jugada_estratosferica", ["epic_play_metadata"]
    if content_type in {"poll", "question", "prediction", "ranking"}:
        return "comunidad_decide", [f"content_type:{content_type}"]
    if content_type == "owned_tournament":
        return "torneo_estratosferico", ["owned_tournament"]
    if content_type == "broadcast_opportunity":
        return "arena_estratosferica", ["broadcast_opportunity"]
    if content_type == "audience_report":
        return "informe_de_audiencia", ["audience_report"]
    if _contains_any(text, SETUP_TERMS) and _contains_any(
        text, BUYING_TERMS
    ):
        return "setup_real", ["technology_purchase_decision"]
    if _contains_any(text, LIFESTYLE_TERMS) and _contains_any(
        text, GAMER_CONNECTION_TERMS
    ):
        return "cultura_en_modo_gamer", ["explicit_gamer_crossover"]
    if (
        str(data.get("territory", "")) == "ai_innovation_future"
        or (
            _contains_any(text, TECH_CHANGE_TERMS)
            and _contains_any(text, TECH_IMPACT_TERMS)
        )
    ):
        matched.append("technology_with_practical_impact")
        return "esto_cambia_el_juego", matched
    return "radar_estratosferico", ["verified_news_default"]


def _rights_state(
    data: Mapping[str, Any], product_id: str
) -> tuple[str, str, bool]:
    rights = data.get("rights")
    if isinstance(rights, Mapping):
        raw_state = str(rights.get("state", "")).strip()
        if raw_state == "link_only_unverified":
            return (
                "official_embed_or_link",
                "Solo enlace o inserción oficial; descarga y republicación "
                "permanecen bloqueadas.",
                True,
            )
        if raw_state in {
            "original_owned",
            "authorized_free",
            "licensed_paid",
            "official_embed_or_link",
        }:
            return raw_state, "Estado de derechos declarado en el radar.", True

    if product_id == "arena_estratosferica":
        return (
            "unverified_blocked",
            "No puede transmitirse sin autorización o licencia documentada.",
            False,
        )
    if product_id == "torneo_estratosferico":
        return (
            "unverified_blocked",
            "Faltan reglamento y constancia de derechos propios.",
            False,
        )
    return (
        "original_owned",
        "La pieza debe usar producción original y citar la fuente; no puede "
        "reutilizar medios de terceros.",
        True,
    )


def classify_candidate(
    candidate: Candidate | Mapping[str, Any],
    strategy: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_strategy(strategy)
    data = _candidate_data(candidate)
    product_id, matched_rules = _select_product(data)
    products = {
        str(product["id"]): product
        for product in strategy["content_products"]
    }
    product = products[product_id]
    rights_state, rights_note, rights_ready = _rights_state(data, product_id)
    decision = {
        "content_product_id": product_id,
        "content_product_name": product["name"],
        "funnel_stage": product["funnel_stage"],
        "purpose": product["purpose"],
        "audience_hypothesis": PRODUCT_AUDIENCE_HYPOTHESES[product_id],
        "expected_community_action": product["community_action"],
        "primary_metric": product["primary_metrics"][0],
        "commercial_path": product["commercial_paths"][0],
        "rights_state": rights_state,
        "rights_note": rights_note,
        "rights_ready_for_draft": rights_ready,
        "matched_rules": matched_rules,
        "classification_mode": "deterministic_strategy_v1",
        "requires_human_review": True,
        "external_actions_enabled": False,
        "publishing_enabled": False,
        "broadcasting_enabled": False,
        "automatic_commercial_outreach_enabled": False,
    }
    errors = validate_strategy_decision(decision)
    if errors:
        raise StrategyConfigurationError(
            f"Decisión estratégica incompleta: {', '.join(errors)}"
        )
    return decision
