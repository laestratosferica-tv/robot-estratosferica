from __future__ import annotations

from .models import Candidate, ContentPackage, Storyboard, StoryboardScene


SOURCE_LABELS = {
    "xbox_wire_es_latam": "Xbox Wire en Español",
    "riot_games_latam": "Riot Games LATAM",
    "esports_insider": "Esports Insider",
    "sportspro": "SportsPro",
    "think_with_google": "Think with Google",
}


def build_storyboard(
    candidate: Candidate,
    package: ContentPackage | None,
) -> Storyboard | None:
    if package is None:
        return None
    question = str(
        package.audience_experiment.get("learning_question", "")
    ).strip()
    scenes = [
        StoryboardScene(
            scene_id="hook",
            start_second=0,
            end_second=3,
            purpose="Detener el scroll con la señal principal.",
            voiceover=candidate.title,
            on_screen_text=candidate.title,
            visual_direction=(
                "Animación tipográfica original basada únicamente en el "
                "titular; sin logos ni material de terceros."
            ),
            audio_direction="Golpe breve y pulso electrónico original.",
        ),
        StoryboardScene(
            scene_id="context",
            start_second=3,
            end_second=8,
            purpose="Presentar únicamente el hecho confirmado.",
            voiceover=str(package.content_punch["short_video_context"]),
            on_screen_text="Hecho confirmado",
            visual_direction=(
                "Composición tipográfica original del resumen factual, sin "
                "representar elementos no mencionados por la fuente."
            ),
            audio_direction="Base rítmica baja; voz completamente legible.",
        ),
        StoryboardScene(
            scene_id="mechanism",
            start_second=8,
            end_second=13,
            purpose="Separar hechos de interpretación.",
            voiceover=(
                "Lectura editorial: esta pieza separa el hecho confirmado "
                "de sus posibles consecuencias."
            ),
            on_screen_text="Lectura editorial",
            visual_direction=(
                "Transición original entre dos bloques rotulados Hecho e "
                "Interpretación."
            ),
            audio_direction="Transición ascendente sutil.",
        ),
        StoryboardScene(
            scene_id="latam_angle",
            start_second=13,
            end_second=20,
            purpose="Añadir la lectura propia de La Estratosférica.",
            voiceover=package.angle,
            on_screen_text="Consecuencias por evaluar",
            visual_direction=(
                "Gráfico abstracto original con signos de pregunta; no añadir "
                "lugares, actores ni cifras ausentes de la evidencia."
            ),
            audio_direction="Mantener ritmo; pausa antes de la pregunta.",
        ),
        StoryboardScene(
            scene_id="why_it_matters",
            start_second=20,
            end_second=26,
            purpose="Recordar el límite de la evidencia.",
            voiceover=(
                "Lectura editorial: cualquier consecuencia debe comprobarse "
                "a partir de la evidencia disponible."
            ),
            on_screen_text="Sin completar vacíos",
            visual_direction=(
                "Subrayar visualmente la fuente y el resumen confirmado."
            ),
            audio_direction="Acento sonoro al completar la conexión.",
        ),
        StoryboardScene(
            scene_id="closing",
            start_second=26,
            end_second=30,
            purpose="Cerrar con una pregunta que genere conversación útil.",
            voiceover=question,
            on_screen_text=question,
            visual_direction=(
                "Cierre tipográfico original, identificación editorial de "
                "La Estratosférica y referencia textual de la fuente."
            ),
            audio_direction="Resolver la música y dejar medio segundo de silencio.",
        ),
    ]
    return Storyboard(
        state="draft",
        master_format="1080x1920",
        duration_seconds=30,
        frames_per_second=30,
        captions_required=True,
        visual_style=[
            "Gráficos, tipografía e ilustraciones originales.",
            "Contraste alto y lectura móvil.",
            "Zona segura para subtítulos y controles de plataforma.",
            "Sin logos, fotografías ni videos de terceros sin aprobación.",
        ],
        scenes=scenes,
        source_card={
            "label": (
                f"Fuente: {SOURCE_LABELS.get(candidate.source_id, 'fuente original')}"
            ),
            "url": candidate.source_url,
        },
    )
