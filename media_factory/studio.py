from __future__ import annotations

from .models import (
    Candidate,
    CommercialOpportunity,
    ContentPackage,
    EditorialDecision,
)


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
    headline = f"{format_id.replace('_', ' ').title()}: {candidate.title}"
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
            f"{headline}\n\n{factual_summary}\n\n{angle}\n\n"
            "¿Lo ves funcionando en Latinoamérica?"
        ),
        "facebook": (
            f"{headline}\n\n{factual_summary} {angle}\n\n"
            "Queremos entender el cambio, no repetir el anuncio."
        ),
        "youtube": (
            f"{headline}\n\n{factual_summary}\n\n"
            f"En este análisis: {angle}\nFuente incluida en la descripción."
        ),
        "threads": (
            f"{candidate.title}. {factual_summary} La pregunta para "
            "Latinoamérica no es cómo copiarlo, sino qué problema "
            "podría resolver aquí."
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
            "Usar gráficos, tipografía e ilustración originales.",
            "No descargar ni reutilizar fotos o videos de la fuente.",
            "Mostrar la fuente como referencia textual.",
            "Formato maestro vertical 1080x1920, adaptable a 1:1 y 16:9.",
        ],
        sources=[candidate.source_url],
        talent=talent or {},
    )
