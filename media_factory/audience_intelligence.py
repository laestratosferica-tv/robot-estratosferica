from __future__ import annotations

import hashlib
from typing import Any

from .models import Candidate, CommercialOpportunity


PLATFORM_PLAYBOOKS: dict[str, dict[str, Any]] = {
    "threads": {
        "formats": ["text_question", "image_context", "short_video"],
        "strength": "conversation",
        "native_poll_api": False,
        "manual_poll_surface": True,
        "metrics": ["views", "likes", "replies", "reposts", "quotes"],
    },
    "instagram": {
        "formats": ["reel", "carousel", "image", "story_question"],
        "strength": "discovery_and_saves",
        "native_poll_api": False,
        "manual_poll_surface": True,
        "metrics": [
            "reach",
            "plays",
            "watch_time",
            "completion_rate",
            "shares",
            "saves",
            "comments",
            "follows",
        ],
    },
    "facebook": {
        "formats": ["reel", "video", "photo_discussion", "link_context"],
        "strength": "community_and_monetization",
        "native_poll_api": False,
        "manual_poll_surface": True,
        "metrics": [
            "reach",
            "views",
            "watch_time",
            "shares",
            "comments",
            "reactions",
            "follows",
            "revenue",
        ],
    },
    "youtube": {
        "formats": ["short", "analysis_video", "community_question"],
        "strength": "authority_and_retention",
        "native_poll_api": False,
        "manual_poll_surface": True,
        "metrics": [
            "views",
            "watch_time",
            "average_percentage_viewed",
            "comments",
            "shares",
            "subscribers_gained",
        ],
    },
}

QUESTION_BANK = {
    "gaming_esports": [
        "¿Qué te mueve más: competir, aprender, pertenecer o ganar premios?",
        "¿Qué juego debería recibir más cobertura competitiva en Latinoamérica?",
        "¿Prefieres resultados rápidos o análisis que expliquen lo que cambió?",
        "¿Qué hace que sigas un torneo: el equipo, el jugador, la historia o el premio?",
    ],
    "sport_technology_entertainment": [
        "¿La tecnología mejora el espectáculo o lo vuelve demasiado técnico?",
        "¿Qué dato sí te ayuda a disfrutar más una competencia?",
        "¿Verías más contenido de rendimiento, negocio, innovación o entretenimiento?",
        "¿Prefieres una explicación corta o una comparación con datos?",
    ],
    "ai_innovation_future": [
        "¿La IA ya te ahorra tiempo o todavía te complica el trabajo?",
        "¿Qué quieres entender mejor: herramientas, empleo, creatividad o negocios?",
        "¿Probarías esta tecnología hoy o esperarías a que madure?",
        "¿Te sirve más un tutorial, una prueba real, una comparación o una opinión?",
    ],
    "brands_activations": [
        "¿Qué te hace recordar una marca: utilidad, emoción, premio o experiencia?",
        "¿Esto se siente útil para la comunidad o solo como publicidad?",
        "¿Qué formato de marca aceptarías: prueba, evento, descuento o contenido?",
        "¿Comprarías por una recomendación si muestra resultados y límites reales?",
    ],
}

OPTION_BANK = {
    "gaming_esports": [
        ["Competir", "Aprender", "Pertenecer", "Premios"],
        ["Fútbol", "Shooter", "MOBA", "Mobile"],
        ["Resultados rápidos", "Análisis con contexto"],
        ["Equipo", "Jugador", "Historia", "Premio"],
    ],
    "sport_technology_entertainment": [
        ["Mejora el show", "Lo vuelve técnico", "Depende del uso"],
        ["Rendimiento", "Estrategia", "Historia", "Ninguno"],
        ["Rendimiento", "Negocio", "Innovación", "Entretenimiento"],
        ["Explicación corta", "Comparación con datos"],
    ],
    "ai_innovation_future": [
        ["Me ahorra tiempo", "Me complica", "Todavía no la uso"],
        ["Herramientas", "Empleo", "Creatividad", "Negocios"],
        ["La probaría", "Esperaría", "Depende del precio"],
        ["Tutorial", "Prueba real", "Comparación", "Opinión"],
    ],
    "brands_activations": [
        ["Utilidad", "Emoción", "Premio", "Experiencia"],
        ["Útil para la comunidad", "Solo publicidad", "Depende"],
        ["Prueba", "Evento", "Descuento", "Contenido"],
        ["Sí, con evidencia", "No", "Depende de quién"],
    ],
}

FORMAT_ROTATIONS = {
    "threads": ["text_question", "image_context", "text_question", "short_video"],
    "instagram": ["reel", "carousel", "image", "carousel"],
    "facebook": ["reel", "photo_discussion", "video", "photo_discussion"],
    "youtube": ["short", "analysis_video", "short", "community_question"],
}

CONTEXT_KEYWORDS = {
    "gaming_esports": [
        {"competir", "competencia", "premio", "torneo", "liga", "equipo"},
        {"juego", "videojuego", "cobertura", "escena", "latinoamérica"},
        {"resultado", "análisis", "explica", "cambió"},
        {"equipo", "jugador", "historia", "premio", "torneo"},
    ],
    "sport_technology_entertainment": [
        {"tecnología", "espectáculo", "show", "experiencia"},
        {"dato", "estadística", "rendimiento", "medición"},
        {"rendimiento", "negocio", "innovación", "entretenimiento"},
        {"explicación", "comparación", "datos"},
    ],
    "ai_innovation_future": [
        {
            "trabajo",
            "empleo",
            "productividad",
            "automatización",
            "colaboración",
            "interacciones",
            "ahorra",
            "tiempo",
        },
        {"herramienta", "empleo", "creatividad", "negocio"},
        {"lanzamiento", "función", "producto", "disponible", "probar"},
        {"tutorial", "prueba", "comparación", "opinión"},
    ],
    "brands_activations": [
        {"recordar", "marca", "emoción", "premio", "experiencia"},
        {"comunidad", "publicidad", "útil", "utilidad"},
        {"formato", "evento", "descuento", "contenido"},
        {"comprar", "recomendación", "resultados", "evidencia"},
    ],
}

GAME_PASS_QUESTIONS = {
    "meta_horizon": (
        "¿Usarías más Game Pass si viniera incluido con Meta Horizon+?",
        ["Sí", "No", "Depende del precio"],
    ),
    "general": (
        "¿Qué cambio haría más útil Game Pass para ti?",
        ["Mejor catálogo", "Menor precio", "Más juego en nube", "Más dispositivos"],
    ),
}


def _stable_index(seed: str, size: int) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % size


def _contextual_question_index(candidate: Candidate) -> int:
    """Choose the question closest to the verified story, with a stable fallback."""
    text = f"{candidate.title} {candidate.summary}".casefold()
    keyword_groups = CONTEXT_KEYWORDS[candidate.territory]
    scores = [
        sum(1 for keyword in keywords if keyword in text)
        for keywords in keyword_groups
    ]
    best_score = max(scores, default=0)
    if best_score:
        return scores.index(best_score)
    return _stable_index(
        f"{candidate.title}:question",
        len(QUESTION_BANK[candidate.territory]),
    )


def _story_question(candidate: Candidate) -> tuple[str, list[str]]:
    text = f"{candidate.title} {candidate.summary}".casefold()
    if "game pass" in text:
        key = (
            "meta_horizon"
            if "meta" in text and ("horizon+" in text or "horizon plus" in text)
            else "general"
        )
        question, options = GAME_PASS_QUESTIONS[key]
        return question, list(options)

    question_index = _contextual_question_index(candidate)
    return (
        QUESTION_BANK[candidate.territory][question_index],
        list(OPTION_BANK[candidate.territory][question_index]),
    )


def _content_goal(
    candidate: Candidate,
    opportunity: CommercialOpportunity | None,
) -> str:
    signals = candidate.signals
    if float(signals.get("conversation_potential", 0)) >= 0.75:
        return "audience_learning"
    if opportunity:
        return "commercial_insight"
    if float(signals.get("explanatory_value", 0)) >= 0.75:
        return "authority"
    return "discovery"


def build_audience_experiment(
    candidate: Candidate,
    opportunity: CommercialOpportunity | None = None,
) -> dict[str, Any]:
    """Prepare one measurable community experiment; never publish it."""
    question, answer_options = _story_question(candidate)
    experiment_id = hashlib.sha256(
        f"{candidate.territory}:{candidate.title}:{question}".encode("utf-8")
    ).hexdigest()[:16]
    goal = _content_goal(candidate, opportunity)

    platform_plans: dict[str, dict[str, Any]] = {}
    for platform, rotation in FORMAT_ROTATIONS.items():
        format_id = rotation[
            _stable_index(f"{experiment_id}:{platform}", len(rotation))
        ]
        playbook = PLATFORM_PLAYBOOKS[platform]
        platform_plans[platform] = {
            "format": format_id,
            "strength": playbook["strength"],
            "question": question,
            "answer_options": list(answer_options),
            "native_poll_api": playbook["native_poll_api"],
            "manual_poll_surface": playbook["manual_poll_surface"],
            "poll_fallback": "question_with_structured_options",
            "metrics": list(playbook["metrics"]),
            "state": "draft",
            "publishing_enabled": False,
        }

    return {
        "experiment_id": experiment_id,
        "primary_goal": goal,
        "learning_question": question,
        "answer_options": list(answer_options),
        "hypothesis": (
            "Una pregunta concreta y fácil de responder generará señales útiles "
            "sobre intereses, formatos y afinidad comercial de la comunidad."
        ),
        "platform_plans": platform_plans,
        "comparison_window_hours": 48,
        "minimum_sample_warning": 30,
        "requires_human_review": True,
        "publishing_enabled": False,
    }
