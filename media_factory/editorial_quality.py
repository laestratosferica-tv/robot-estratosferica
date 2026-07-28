from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


VISUAL_METADATA_TERMS = {
    "alt",
    "background",
    "banner",
    "caption",
    "cover",
    "cuadricula",
    "descripcion",
    "fondo",
    "foto",
    "fotografia",
    "grid",
    "ilustracion",
    "image",
    "imagen",
    "logo",
    "logotipo",
    "main",
    "persona",
    "photo",
    "picture",
    "principal",
    "screen",
    "screenshot",
    "texto",
    "thumbnail",
    "visible",
}

VISUAL_ONLY_PREDICATES = {
    "aparece",
    "aparecen",
    "contains",
    "contiene",
    "dice",
    "features",
    "muestra",
    "shows",
}

INFORMATIVE_PREDICATES = {
    "added",
    "adds",
    "analiza",
    "analizan",
    "anuncia",
    "anunciaron",
    "anuncio",
    "announced",
    "announces",
    "aumenta",
    "aumento",
    "cambia",
    "changed",
    "changes",
    "confirma",
    "confirmo",
    "confirmed",
    "documenta",
    "documents",
    "explica",
    "explains",
    "incluye",
    "includes",
    "incorpora",
    "incorporates",
    "integra",
    "integrates",
    "lanza",
    "launched",
    "launches",
    "llega",
    "permite",
    "publica",
    "published",
    "reaches",
    "reduce",
    "revela",
    "reveals",
    "suma",
    "updates",
    "works",
}

TEMPORAL_OR_QUANTITY_TERMS = {
    "ayer",
    "desde",
    "durante",
    "hoy",
    "manana",
    "million",
    "millones",
    "month",
    "paises",
    "porcentaje",
    "semanas",
    "today",
    "tomorrow",
    "week",
}

NON_FACTUAL_FILLER = {
    "a",
    "al",
    "and",
    "con",
    "de",
    "del",
    "el",
    "en",
    "for",
    "la",
    "las",
    "los",
    "of",
    "on",
    "para",
    "por",
    "que",
    "sobre",
    "the",
    "todo",
    "una",
    "un",
    "y",
}

PLACEHOLDER_TERMS = {
    "na",
    "none",
    "pendiente",
    "placeholder",
    "prueba",
    "tbd",
    "test",
}

CONTEXT_DOMAINS = {
    "alliance": {
        "alianza",
        "alianzas",
        "alliance",
        "partnership",
    },
    "product_platform": {
        "plataforma",
        "plataformas",
        "platform",
        "platforms",
        "producto",
        "productos",
        "product",
        "products",
    },
    "gaming": {
        "gamer",
        "gamers",
        "gaming",
        "juega",
        "juego",
        "juegos",
        "jugador",
        "jugadores",
        "play",
        "player",
        "players",
        "videojuego",
        "videojuegos",
    },
    "competition": {
        "competencia",
        "competir",
        "competitive",
        "compete",
        "esports",
        "torneo",
        "tournament",
    },
}


def normalize_editorial_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    without_marks = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def text_is_equivalent(left: str, right: str) -> bool:
    """Detect lexical equivalence without models or external calls."""
    normalized_left = normalize_editorial_text(left)
    normalized_right = normalize_editorial_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True

    left_tokens = normalized_left.split()
    right_tokens = normalized_right.split()
    if min(len(left_tokens), len(right_tokens)) < 4:
        return False

    sequence_similarity = SequenceMatcher(
        None,
        normalized_left,
        normalized_right,
    ).ratio()
    shared_tokens = set(left_tokens) & set(right_tokens)
    token_coverage = len(shared_tokens) / max(
        len(set(left_tokens)),
        len(set(right_tokens)),
    )
    length_ratio = min(len(left_tokens), len(right_tokens)) / max(
        len(left_tokens),
        len(right_tokens),
    )
    return (
        sequence_similarity >= 0.92
        or (token_coverage >= 0.90 and length_ratio >= 0.80)
    )


def _tokens(value: str) -> list[str]:
    return normalize_editorial_text(value).split()


def _is_placeholder(summary: str) -> bool:
    tokens = _tokens(summary)
    if not tokens:
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if token in PLACEHOLDER_TERMS:
            return True
        if len(token) <= 4 and len(set(token)) == 1:
            return True
    return False


def _is_visual_metadata_only(title: str, summary: str) -> bool:
    tokens = _tokens(summary)
    token_set = set(tokens)
    if not token_set.intersection(VISUAL_METADATA_TERMS):
        return False
    has_informative_predicate = bool(
        token_set.intersection(INFORMATIVE_PREDICATES)
    )
    has_numeric_fact = bool(
        set(re.findall(r"\d+(?:[.,]\d+)?", summary))
        - set(re.findall(r"\d+(?:[.,]\d+)?", title))
    )
    has_quantity_or_time = bool(
        token_set.intersection(TEMPORAL_OR_QUANTITY_TERMS)
    )
    return not (
        has_informative_predicate
        or has_numeric_fact
        or has_quantity_or_time
    )


def _has_distinct_informative_proposition(title: str, summary: str) -> bool:
    summary_tokens = _tokens(summary)
    title_tokens = set(_tokens(title))
    if len(summary_tokens) < 4:
        return False

    content_tokens = {
        token
        for token in summary_tokens
        if token not in NON_FACTUAL_FILLER
        and token not in VISUAL_METADATA_TERMS
        and token not in VISUAL_ONLY_PREDICATES
    }
    if len(content_tokens) < 3:
        return False

    distinct_tokens = content_tokens - title_tokens
    has_numeric_fact = bool(re.search(r"\d", summary))
    has_quantity_or_time = bool(
        content_tokens.intersection(TEMPORAL_OR_QUANTITY_TERMS)
    )
    has_predicate = bool(
        content_tokens.intersection(INFORMATIVE_PREDICATES)
    )
    has_proposition_shape = (
        has_predicate
        or has_numeric_fact
        or has_quantity_or_time
        or len(content_tokens) >= 6
    )
    has_distinct_information = (
        len(distinct_tokens) >= 2
        or has_numeric_fact
        or has_quantity_or_time
    )
    return has_proposition_shape and has_distinct_information


def substantive_summary_issue(title: str, summary: str) -> str | None:
    if not normalize_editorial_text(summary):
        return "missing_substantive_summary"
    if text_is_equivalent(title, summary):
        return "summary_equivalent_to_title"
    if _is_placeholder(summary):
        return "summary_placeholder"
    if _is_visual_metadata_only(title, summary):
        return "summary_visual_metadata_only"
    if not _has_distinct_informative_proposition(title, summary):
        return "summary_lacks_distinct_informative_proposition"
    return None


def unsupported_context_domains(
    evidence: str,
    generated_text: str,
) -> list[str]:
    """Report domain frames introduced by generated copy but absent in evidence."""
    evidence_tokens = set(_tokens(evidence))
    generated_tokens = set(_tokens(generated_text))
    return [
        name
        for name, terms in CONTEXT_DOMAINS.items()
        if generated_tokens.intersection(terms)
        and not evidence_tokens.intersection(terms)
    ]
