from __future__ import annotations

from .models import (
    Candidate,
    CommercialOpportunity,
    ContentPackage,
    EditorialDecision,
)
from .audience_intelligence import build_audience_experiment
from .content_punch import build_content_punch


FORMAT_BY_TERRITORY = {
    "gaming_esports": "radar_estratosferico",
    "sport_technology_entertainment": "esto_cambia_el_juego",
    "ai_innovation_future": "esto_cambia_el_juego",
    "brands_activations": "brand_play",
}

ANGLE_BY_FORMAT = {
    "radar_estratosferico": (
        "La clave es entender por qué esta señal importa para la cultura gamer "
        "latinoamericana."
    ),
    "esto_cambia_el_juego": (
        "El valor está en traducir el cambio tecnológico en una consecuencia "
        "concreta "
        "para audiencias y negocios."
    ),
    "brand_play": (
        "La lectura útil está en identificar qué problema resuelve la "
        "activación y qué podría "
        "adaptarse a Latinoamérica sin copiarla."
    ),
}


def _sentence(text: str) -> str:
    value = text.strip()
    if not value:
        return value
    return value if value[-1] in ".!?…" else f"{value}."


def _trim_at_word(text: str, limit: int) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    if limit <= 1:
        return "…"[:limit]
    shortened = value[: limit - 1].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return f"{shortened.rstrip(' ,;:-')}…"


def _threads_copy(
    headline: str,
    concrete_value: str,
    tension_question: str,
    limit: int = 500,
) -> str:
    head = _sentence(headline)
    value = _sentence(concrete_value)
    question = tension_question.strip()
    rendered = f"{head} {value} {question}"
    if len(rendered) <= limit:
        return rendered

    available_before_question = limit - len(question) - 2
    if available_before_question < 20:
        return _trim_at_word(rendered, limit)
    head_limit = min(len(headline), max(80, available_before_question // 2))
    head = _sentence(_trim_at_word(headline, head_limit))
    value_limit = limit - len(head) - len(question) - 2
    value = _sentence(_trim_at_word(concrete_value, value_limit))
    return f"{head} {value} {question}"


def build_content_package(
    candidate: Candidate,
    decision: EditorialDecision,
    opportunity: CommercialOpportunity | None,
    talent: dict[str, str] | None = None,
) -> ContentPackage | None:
    if not decision.accepted:
        return None
    format_id = FORMAT_BY_TERRITORY[candidate.territory]
    angle = ANGLE_BY_FORMAT[format_id]
    factual_summary = candidate.summary or (
        f"La fuente seleccionada presenta la historia: {candidate.title}."
    )
    audience_experiment = build_audience_experiment(candidate, opportunity)
    content_punch = build_content_punch(candidate, audience_experiment)
    headline = content_punch["hook"]
    commercial_line = (
        "El valor no está solo en el anuncio: está en cómo conecta "
        "producto, plataforma y hábito de consumo."
        if opportunity
        else "Su valor está en lo que revela sobre los cambios de la audiencia."
    )
    script = (
        f"GANCHO\n{candidate.title}\n\n"
        f"CONTEXTO\n{factual_summary}\n\n"
        f"LECTURA ESTRATOSFÉRICA\n{angle}\n\n"
        f"POR QUÉ IMPORTA\n{commercial_line}\n\n"
        "CIERRE\n¿Qué tendría que cambiar para que esta idea funcione "
        "de verdad en Latinoamérica?"
    )
    platform_copy = {
        "instagram": (
            f"{headline}\n\n{content_punch['concrete_value']}\n\n"
            f"{content_punch['tension_question']}\n"
            f"{content_punch['expected_action']}"
        ),
        "facebook": (
            f"{headline}\n\n{content_punch['concrete_value']}\n\n"
            f"{content_punch['audience_promise']}\n\n"
            f"{content_punch['tension_question']}\n"
            f"{content_punch['expected_action']}"
        ),
        "youtube": (
            f"{headline}\n\n{content_punch['concrete_value']}\n\n"
            f"{content_punch['audience_promise']}\n"
            f"{content_punch['tension_question']}\n"
            "Fuente incluida en la descripción."
        ),
        "threads": (
            _threads_copy(
                headline,
                content_punch["concrete_value"],
                content_punch["tension_question"],
            )
        ),
    }
    return ContentPackage(
        format_id=format_id,
        state="draft",
        headline=headline,
        angle=angle,
        factual_summary=factual_summary,
        short_video_script=script,
        platform_copy=platform_copy,
        visual_brief=[
            f"Gancho dominante: {content_punch['hook']}",
            f"Valor concreto visible: {content_punch['concrete_value']}",
            (
                "Pregunta o tensión visible: "
                f"{content_punch['tension_question']}"
            ),
            f"Acción esperada: {content_punch['expected_action']}",
            f"Energía visual: {content_punch['visual_energy']}",
            "Usar gráficos, tipografía e ilustración originales.",
            "No descargar ni reutilizar fotos o videos de la fuente.",
            "Mostrar la fuente como referencia textual.",
            "Formato maestro vertical 1080x1920, adaptable a 1:1 y 16:9.",
        ],
        sources=[candidate.source_url],
        talent=talent or {},
        audience_experiment=audience_experiment,
        content_punch=content_punch,
    )
