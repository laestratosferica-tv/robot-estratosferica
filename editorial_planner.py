import random


GAMER_CTAS = [
    "¿W o humo?",
    "¿Skill o regalo?",
    "¿Clutch o suerte?",
    "¿Banco o no?",
    "¿Esto cuenta o no?",
]

PILLAR_CTAS = {
    "gaming": GAMER_CTAS,
    "technology": ["¿Lo usarías o pasas?", "¿Upgrade o puro humo?", "¿Esto sí cambia el juego?"],
    "advertising": ["¿Campañón o humazo?", "¿Te vendería o no?", "¿Idea brutal o reciclada?"],
    "fashion": ["¿Te lo pondrías?", "¿Drop o puro hype?", "¿Estilo o disfraz?"],
    "gastronomy": ["¿Lo probarías?", "¿Experiencia o puro show?", "¿Se antoja o no?"],
    "lifestyle": ["¿Lo sumarías a tu vida?", "¿Tendencia útil o pose?", "¿Sí mejora la experiencia?"],
    "luxury": ["¿Lujo real o solo precio?", "¿Objeto de deseo o exageración?", "¿Lo vale?"],
    "monetization": ["¿Oportunidad o humo?", "¿Pagarías por esto?", "¿Negocio real o promesa?"],
}

PILLAR_BADGES = {
    "technology": "TECH",
    "advertising": "CREA",
    "fashion": "STYLE",
    "gastronomy": "FOOD",
    "lifestyle": "LIFE",
    "luxury": "LUX",
    "monetization": "NEGOCIO",
}

PILLAR_TITLES = {
    "technology": ["ESTO CAMBIA EL JUEGO", "¿UPGRADE O HUMO?", "NUEVA HERRAMIENTA"],
    "advertising": ["¿CAMPAÑÓN O HUMO?", "ESTA IDEA SÍ VENDE", "MIRA LA JUGADA"],
    "fashion": ["EL DROP DEL MOMENTO", "GAMING SE VISTE ASÍ", "¿ESTILO O HYPE?"],
    "gastronomy": ["ESTO SE COME CON LOS OJOS", "LA EXPERIENCIA CAMBIÓ", "¿ANTOJO O HYPE?"],
    "lifestyle": ["ASÍ VIVE EL TARGET DEL FUTURO", "ESTO YA ES CULTURA", "¿TENDENCIA O CAMBIO REAL?"],
    "luxury": ["EL NUEVO LUJO ES DIGITAL", "¿LUJO O EXCESO?", "OBJETO DE DESEO"],
    "monetization": ["¿NEGOCIO O HUMO?", "AQUÍ HAY OPORTUNIDAD", "¿PAGARÍAS POR ESTO?"],
}


def _clean(text):
    return (text or "").strip()


def _has_any(text, words):
    t = _clean(text).lower()
    return any(w in t for w in words)


def pick_cta_by_style(style_family, default_cta=None, pillar="gaming"):
    if default_cta and default_cta.strip() and default_cta.strip().lower() != "sigue para más":
        return default_cta.strip()
    return random.choice(PILLAR_CTAS.get(pillar, GAMER_CTAS))


def pick_badge_by_title(title, pillar="gaming"):
    t = _clean(title).lower()
    if pillar in PILLAR_BADGES:
        return PILLAR_BADGES[pillar]

    if _has_any(t, ["final", "grand final", "grand finals", "playoffs", "masters", "worlds", "champion", "campeón"]):
        return "FINAL"
    if _has_any(t, ["bug", "exploit", "glitch"]):
        return "BUG"
    if _has_any(t, ["buff", "nerf", "meta", "patch", "update", "parche"]):
        return "META"
    if _has_any(t, ["leak", "filtrado", "rumor"]):
        return "LEAK"
    if _has_any(t, ["ace", "pentakill", "quadrakill", "quadra", "clutch"]):
        return "CLIP"

    return random.choice(["HOT", "PLAY", "TOP"])


def build_reel_gamer_title(headline):
    t = (headline or "").lower()

    if "valorant" in t and _has_any(t, ["final", "masters", "champions", "playoffs"]):
        return "TODO O NADA"
    if "valorant" in t and _has_any(t, ["ace", "clutch"]):
        return "CLUTCH O HUMO"
    if _has_any(t, ["league of legends", "lol"]) and _has_any(t, ["quadra", "quadrakill", "penta", "pentakill"]):
        return "NO ES NORMAL"
    if _has_any(t, ["league of legends", "lol"]) and _has_any(t, ["final", "worlds", "playoffs"]):
        return "ESTO SE CALENTÓ"
    if _has_any(t, ["bug", "exploit", "glitch"]):
        return "BUG O SKILL"
    if _has_any(t, ["record", "récord"]):
        return "HISTORIA O SUERTE"
    if _has_any(t, ["ace", "pentakill", "quadrakill", "quadra", "clutch"]):
        return "NO TIENE SENTIDO"
    if _has_any(t, ["final", "grand final", "grand finals", "playoffs", "masters", "worlds"]):
        return "MOMENTO CLAVE"

    return random.choice([
        "¿QUÉ ACABO DE VER?",
        "ESTO NO ES REAL",
        "ALGO PASÓ AQUÍ",
        "NO TIENE SENTIDO",
        "MIRA ESTO",
        "SE VOLVIÓ LOCO",
    ])


def build_visual_title(headline, pillar="gaming"):
    if pillar in PILLAR_TITLES:
        return random.choice(PILLAR_TITLES[pillar])
    return build_reel_gamer_title(headline)


def choose_style_family(title, pillar="gaming"):
    return {
        "gaming": "reel_gamer",
        "technology": "reel_tech",
        "advertising": "reel_creative",
        "fashion": "reel_style",
        "gastronomy": "reel_food",
        "lifestyle": "reel_life",
        "luxury": "reel_luxury",
        "monetization": "reel_opportunity",
    }.get(pillar, "reel_gamer")


def should_use_runway(style_family, runway_enabled, runway_force):
    if not runway_enabled:
        return False
    if runway_force:
        return True
    return random.random() <= 0.35


def build_runway_prompt(title, style_family):
    return (
        "High-energy digital culture motion background, modern Latin creator aesthetic, "
        "premium editorial composition, subtle motion and social-first energy, "
        "not news style, not corporate flyer. Style: " + style_family +
        ". Headline context: " + (title or "")
    )


def build_editorial_plan(item, default_cta=None, runway_enabled=False, runway_force=False):
    raw_title = item.get("title", "") or ""
    pillar = item.get("pillar", "gaming")
    style_family = choose_style_family(raw_title, pillar)
    title_text = build_visual_title(raw_title, pillar)
    cta_text = pick_cta_by_style(style_family, default_cta, pillar)
    badge_text = pick_badge_by_title(raw_title, pillar)
    use_runway = should_use_runway(style_family, runway_enabled, runway_force)
    runway_prompt = build_runway_prompt(raw_title, style_family)
    return {
        "pillar": pillar,
        "style_family": style_family,
        "title_text": title_text,
        "cta_text": cta_text,
        "badge_text": badge_text,
        "use_runway": use_runway,
        "runway_prompt": runway_prompt,
        "motion_level": "high" if pillar in {"gaming", "technology", "advertising"} else "medium",
    }
