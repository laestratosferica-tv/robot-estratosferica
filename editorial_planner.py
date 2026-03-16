import random


GAMER_CTAS = [
    "¿W o humo?",
    "¿Lo compras o no?",
    "¿Roto o normal?",
    "¿Te gusta o puro show?",
    "¿Esto suma o estorba?",
    "¿Banco o no banco?",
]


def _clean(text):
    return (text or "").strip()


def _has_any(text, words):
    t = _clean(text).lower()
    return any(w in t for w in words)


def pick_cta_by_style(style_family, default_cta=None):
    if default_cta and default_cta.strip() and default_cta.strip().lower() != "sigue para más":
        return default_cta.strip()

    if style_family == "reel_gamer":
        return random.choice([
            "¿W o humo?",
            "¿Esto está roto?",
            "¿Banco o no banco?",
            "¿Lo ves o no?",
            "¿Skill o regalo?",
        ])

    return random.choice(GAMER_CTAS)


def pick_badge_by_title(title):
    t = _clean(title).lower()

    if _has_any(t, ["leak", "filtra", "filtrado"]):
        return "LEAK"
    if _has_any(t, ["drama", "polémica", "controversia", "funa"]):
        return "DRAMA"
    if _has_any(t, ["parche", "patch", "update", "actualización"]):
        return "UPDATE"
    if _has_any(t, ["final", "champion", "campeón", "worlds", "major"]):
        return "FINAL"
    if _has_any(t, ["buff", "nerf", "meta"]):
        return "META"
    return random.choice(["HOT", "TOP", "OJO"])


def build_visual_title(title, style_family):
    t = _clean(title)

    if not t:
        return "ESTO SE PUSO RARO"

    tl = t.lower()

    if style_family == "reel_gamer":
        if _has_any(tl, ["drama", "polémica", "controversia", "funa"]):
            return random.choice([
                "ESTO YA SE SALIÓ",
                "SE PRENDIÓ TODO",
                "YA HUELE A DRAMA",
                "ESTO VA A DIVIDIR",
            ])
        if _has_any(tl, ["parche", "patch", "buff", "nerf", "meta"]):
            return random.choice([
                "ESTO MUEVE EL META",
                "AQUÍ CAMBIÓ TODO",
                "ESTO NO VIENE SUAVE",
                "SE VIENE PROBLEMA",
            ])
        if _has_any(tl, ["leak", "filtrado", "rumor"]):
            return random.choice([
                "SE LES ESCAPÓ",
                "YA SE FILTRÓ TODO",
                "OJO CON ESTO",
                "ESTO NO ERA OFICIAL",
            ])
        return random.choice([
            "ESTO NO PASÓ NORMAL",
            "OJO CON ESTO",
            "AQUÍ HAY TEMA",
            "ESTO TIENE PINTA",
        ])

    if _has_any(tl, ["parche", "patch", "update", "actualización"]):
        return random.choice([
            "CAMBIO IMPORTANTE",
            "NUEVO UPDATE",
            "AJUSTE CLAVE",
        ])
    if _has_any(tl, ["final", "major", "worlds", "campeón"]):
        return random.choice([
            "MOMENTO CLAVE",
            "SE VIENE LO SERIO",
            "PUNTO CRÍTICO",
        ])

    return t[:55].upper()


def choose_style_family(title):
    return "reel_gamer"

def should_use_runway(style_family, runway_enabled, runway_force):
    if not runway_enabled:
        return False
    if runway_force:
        return True

    if style_family == "editorial_clean":
        return random.random() <= 0.45

    return random.random() <= 0.15


def build_runway_prompt(title, style_family):
    if style_family == "reel_gamer":
        return (
            "High-energy gaming motion background, fast camera movement, "
            "subtle glitch, intense esports vibe, punchy, not cinematic trailer, "
            "not news style, social reel energy. Headline context: " + (title or "")
        )

    return (
        "Dynamic gaming editorial motion, subtle camera movement, premium esports feel, "
        "clean but alive, social-first vertical video. Headline context: " + (title or "")
    )


def build_editorial_plan(item, default_cta=None, runway_enabled=False, runway_force=False):
    raw_title = item.get("title", "") or ""

    style_family = choose_style_family(raw_title)
    title_text = build_visual_title(raw_title, style_family)
    cta_text = pick_cta_by_style(style_family, default_cta)
    badge_text = pick_badge_by_title(raw_title)
    use_runway = should_use_runway(style_family, runway_enabled, runway_force)
    runway_prompt = build_runway_prompt(raw_title, style_family)

    return {
        "style_family": style_family,
        "title_text": title_text,
        "cta_text": cta_text,
        "badge_text": badge_text,
        "use_runway": use_runway,
        "runway_prompt": runway_prompt,
        "motion_level": "high" if style_family == "reel_gamer" else "low",
    }
