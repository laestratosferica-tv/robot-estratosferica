"""Hashtags precisos para descubrimiento editorial en gaming LATAM."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable


BRAND_TAG = "#LaEstratosferica"
AUDIENCE_TAG = "#GamingLatam"

GAME_PROFILES = {
    "halo": ("#Halo", "#MasterChief"),
    "valorant": ("#Valorant", "#VCT"),
    "counter strike": ("#CounterStrike", "#CS2"),
    "cs2": ("#CS2", "#CounterStrike"),
    "league of legends": ("#LeagueOfLegends", "#LoL"),
    "fortnite": ("#Fortnite",),
    "warzone": ("#Warzone", "#CallOfDuty"),
    "call of duty": ("#CallOfDuty",),
    "apex legends": ("#ApexLegends",),
    "minecraft": ("#Minecraft",),
    "ea sports fc": ("#EASportsFC",),
    "gran turismo": ("#GranTurismo", "#SimRacing"),
}

INTENT_TAGS = {
    "news": "#NoticiasGaming",
    "noticia": "#NoticiasGaming",
    "launch": "#LanzamientosGaming",
    "lanzamiento": "#LanzamientosGaming",
    "esports": "#EsportsLatam",
    "competitive": "#EsportsLatam",
    "competitivo": "#EsportsLatam",
    "gameplay": "#Gameplay",
}

GENERIC_FILLER = {
    "#fyp",
    "#viral",
    "#parati",
    "#explorepage",
    "#reels",
    "#reelsgaming",
}


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _topic_tag(topic: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", _plain(topic))
    return f"#{''.join(word[:1].upper() + word[1:] for word in words)}" if words else ""


def select_hashtags(
    *,
    game: str = "",
    title: str = "",
    topic: str = "",
    intent: str = "news",
    extra_tags: Iterable[str] = (),
    limit: int = 6,
) -> list[str]:
    """Devuelve 4-6 etiquetas semánticas, sin relleno ni duplicados."""
    context = _plain(f"{game} {title}")
    tags: list[str] = []

    for key, profile_tags in GAME_PROFILES.items():
        if key in context:
            tags.extend(profile_tags)
            break

    topic_tag = _topic_tag(topic)
    if topic_tag:
        tags.append(topic_tag)

    intent_key = _plain(intent)
    tags.append(INTENT_TAGS.get(intent_key, "#NoticiasGaming"))
    tags.extend([AUDIENCE_TAG, BRAND_TAG])
    tags.extend(extra_tags)

    selected: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = str(tag).strip()
        if not clean.startswith("#"):
            clean = f"#{clean}"
        key = clean.casefold()
        if clean.casefold() in GENERIC_FILLER or key in seen:
            continue
        if not re.fullmatch(r"#[A-Za-z0-9_]+", clean):
            continue
        seen.add(key)
        selected.append(clean)
        if len(selected) >= max(1, min(limit, 6)):
            break

    return selected


def replace_hashtags(text: str, hashtags: Iterable[str]) -> str:
    """Quita etiquetas generadas libremente y agrega la selección validada."""
    body = re.sub(r"(?<!\w)#[^\s#]+", "", text or "")
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    suffix = " ".join(hashtags)
    return f"{body}\n\n{suffix}".strip() if suffix else body
