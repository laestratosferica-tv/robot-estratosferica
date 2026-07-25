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
    if package.format_id == "brand_play":
        mechanism_voiceover = (
            "La historia conecta una marca, una plataforma y una nueva "
            "forma de consumo."
        )
        mechanism_text = "La experiencia es el canal"
        why_voiceover = (
            "El valor no está solo en el anuncio: está en cómo conecta "
            "producto, plataforma y hábito."
        )
        why_text = "Más experiencia, menos anuncio"
    else:
        mechanism_voiceover = (
            "La señal conecta cultura digital, comportamiento de audiencia "
            "y un cambio de mercado."
        )
        mechanism_text = "La señal detrás de la noticia"
        why_voiceover = (
            "Importa por lo que revela sobre la forma de jugar, competir "
            "y construir comunidad."
        )
        why_text = "Audiencia + cultura + cambio"
    scenes = [
        StoryboardScene(
            scene_id="hook",
            start_second=0,
            end_second=3,
            purpose="Detener el scroll con la señal principal.",
            voiceover=candidate.title,
            on_screen_text="No es solo una alianza",
            visual_direction=(
                "Animación tipográfica original con dos sistemas que se conectan; "
                "sin logos ni material de terceros."
            ),
            audio_direction="Golpe breve y pulso electrónico original.",
        ),
        StoryboardScene(
            scene_id="context",
            start_second=3,
            end_second=8,
            purpose="Presentar únicamente el hecho confirmado.",
            voiceover=package.factual_summary,
            on_screen_text="Producto + plataforma",
            visual_direction=(
                "Diagrama original de producto, plataforma y usuario con "
                "etiquetas genéricas."
            ),
            audio_direction="Base rítmica baja; voz completamente legible.",
        ),
        StoryboardScene(
            scene_id="mechanism",
            start_second=8,
            end_second=13,
            purpose="Mostrar el mecanismo sin repetir el comunicado.",
            voiceover=mechanism_voiceover,
            on_screen_text=mechanism_text,
            visual_direction=(
                "Tres bloques originales se unen en una ruta visual simple."
            ),
            audio_direction="Transición ascendente sutil.",
        ),
        StoryboardScene(
            scene_id="latam_angle",
            start_second=13,
            end_second=20,
            purpose="Añadir la lectura propia de La Estratosférica.",
            voiceover=package.angle,
            on_screen_text="¿Qué se adapta a LATAM?",
            visual_direction=(
                "Mapa abstracto original de Latinoamérica con nodos de "
                "audiencia, acceso y comunidad."
            ),
            audio_direction="Mantener ritmo; pausa antes de la pregunta.",
        ),
        StoryboardScene(
            scene_id="why_it_matters",
            start_second=20,
            end_second=26,
            purpose="Explicar el valor estratégico.",
            voiceover=why_voiceover,
            on_screen_text=why_text,
            visual_direction=(
                "Sistema original de tres círculos: producto, plataforma y hábito."
            ),
            audio_direction="Acento sonoro al completar la conexión.",
        ),
        StoryboardScene(
            scene_id="closing",
            start_second=26,
            end_second=30,
            purpose="Cerrar con una pregunta que genere conversación útil.",
            voiceover=(
                "¿Qué tendría que cambiar para que esta idea funcione de "
                "verdad en Latinoamérica?"
            ),
            on_screen_text="¿Funcionaría aquí?",
            visual_direction=(
                "Cierre tipográfico original, identificación del formato "
                f"{package.format_id.replace('_', ' ').title()} y referencia "
                "textual de la fuente."
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
