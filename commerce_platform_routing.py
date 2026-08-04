"""Rutas comerciales seguras y adaptadas por plataforma.

Este módulo prepara copys y comprobaciones. No publica contenido ni modifica
credenciales. La activación de cada red sigue bajo la compuerta supervisada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from urllib.parse import parse_qs, urlparse


AFFILIATE_DISCLOSURE = "Como Afiliado de Amazon, recibimos ingresos por compras adscritas."


@dataclass(frozen=True)
class CommerceRoute:
    platform: str
    cta: str
    destination: str
    requires_clickable_link: bool
    publication_format: str


def build_amazon_routes(
    affiliate_url: str,
    *,
    profile_destination: str = "enlace de Amazon en el perfil",
    expected_tag: str | None = None,
    expected_host: str = "www.amazon.com",
) -> Dict[str, CommerceRoute]:
    """Devuelve la ruta de conversión adecuada para cada plataforma.

    Instagram y YouTube Shorts no dependen de una URL escrita en el copy.
    Facebook y Threads conservan el enlace directo. Instagram exige además
    una Story orgánica con sticker para ofrecer una salida pulsable.
    """

    url = affiliate_url.strip()
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("El enlace afiliado debe usar HTTPS")
    if parsed.hostname != expected_host:
        raise ValueError("El enlace afiliado debe pertenecer al marketplace Amazon configurado")
    if expected_tag and parse_qs(parsed.query).get("tag") != [expected_tag]:
        raise ValueError("El enlace afiliado no contiene el associate tag configurado")

    return {
        "instagram_reel": CommerceRoute(
            platform="instagram_reel",
            cta="Mira la opción real en Amazon desde el enlace del perfil.",
            destination=profile_destination,
            requires_clickable_link=False,
            publication_format="reel",
        ),
        "instagram_story": CommerceRoute(
            platform="instagram_story",
            cta="Ver opción real en Amazon",
            destination=url,
            requires_clickable_link=True,
            publication_format="story_link_sticker",
        ),
        "facebook": CommerceRoute(
            platform="facebook",
            cta="Ver opción real en Amazon",
            destination=url,
            requires_clickable_link=True,
            publication_format="native_post_or_reel",
        ),
        "threads": CommerceRoute(
            platform="threads",
            cta="Revisa la opción real en Amazon",
            destination=url,
            requires_clickable_link=True,
            publication_format="native_conversation",
        ),
        "youtube_short": CommerceRoute(
            platform="youtube_short",
            cta="Producto y enlace en el perfil del canal.",
            destination=profile_destination,
            requires_clickable_link=False,
            publication_format="short",
        ),
    }


def validate_amazon_distribution(
    routes: Dict[str, CommerceRoute],
) -> list[str]:
    """Detecta piezas incompletas antes de aprobar una distribución."""

    required = {
        "instagram_reel",
        "instagram_story",
        "facebook",
        "threads",
        "youtube_short",
    }
    errors = []
    missing = sorted(required - set(routes))
    if missing:
        errors.append("Faltan rutas: " + ", ".join(missing))

    for name, route in routes.items():
        if route.requires_clickable_link and not route.destination.startswith("https://"):
            errors.append(f"{name}: falta enlace HTTPS clicable")

    return errors
