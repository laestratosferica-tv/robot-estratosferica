import random
from typing import Any, Dict, List


DEFAULT_WEIGHTS = {
    "gaming": 35,
    "technology": 25,
    "advertising": 10,
    "fashion": 8,
    "gastronomy": 7,
    "lifestyle": 7,
    "luxury": 4,
    "monetization": 4,
}

KEYWORDS = {
    "gaming": [
        "gaming", "game", "esport", "valorant", "league of legends", "fortnite",
        "counter-strike", "cs2", "dota", "free fire", "playstation", "xbox",
        "nintendo", "steam", "twitch", "roblox", "minecraft",
    ],
    "technology": [
        "technology", "tech", "artificial intelligence", " ai ", "openai",
        "google ai", "gemini", "hardware", "gpu", "processor", "software",
        "app", "creator tool", "virtual reality", "cybersecurity", "tecnología",
        "inteligencia artificial", "realidad virtual",
    ],
    "advertising": [
        "advertising", "marketing", "campaign", "brand", "branding", "creative",
        "social media", "content strategy", "audience", "creator economy",
        "publicidad", "campaña", "creatividad", "creadores",
    ],
    "fashion": [
        "fashion", "streetwear", "sneaker", "style", "designer", "runway",
        "collaboration", "drop", "moda", "tenis", "colección", "ropa",
    ],
    "gastronomy": [
        "food", "restaurant", "chef", "drink", "cocktail", "coffee", "culinary",
        "gastronomy", "comida", "restaurante", "gastronomía", "café", "cocina",
    ],
    "lifestyle": [
        "lifestyle", "travel", "wellness", "fitness", "home", "mobility",
        "experience", "culture", "viaje", "bienestar", "hogar", "experiencia",
    ],
    "luxury": [
        "luxury", "premium", "watch", "jewelry", "supercar", "yacht", "hotel",
        "lujo", "reloj", "joyería", "exclusivo", "alta gama",
    ],
    "monetization": [
        "monetization", "affiliate", "subscription", "sponsor", "revenue",
        "business model", "startup", "ecommerce", "product hunt", "creator fund",
        "monetización", "afiliado", "patrocinio", "negocio digital",
    ],
}

DOMAIN_HINTS = {
    "gaming": ["dexerto.com", "pcgamer.com", "gamespot.com", "esports", "dotesports.com", "hltv.org"],
    "technology": ["techcrunch.com", "blog.google", "openai.com", "theverge.com", "wired.com"],
    "advertising": ["hubspot.com", "marketingdirecto.com", "adweek.com"],
    "fashion": ["hypebeast.com", "highsnobiety.com", "vogue.com", "complex.com/style"],
    "gastronomy": ["eater.com", "foodandwine.com", "bonappetit.com"],
    "lifestyle": ["dezeen.com", "travelandleisure.com", "fastcompany.com"],
    "luxury": ["luxurydaily.com", "robbreport.com", "wallpaper.com"],
    "monetization": ["producthunt.com", "shopify.com"],
}

COMMERCIAL_ANGLES = {
    "gaming": ["patrocinio", "torneos", "transmisión_en_vivo", "activación_de_marca"],
    "technology": ["afiliación", "demo_de_producto", "contenido_patronicado", "generación_de_leads"],
    "advertising": ["branded_content", "servicios_creativos", "caso_de_estudio", "consultoría"],
    "fashion": ["drops", "afiliación", "colaboración", "live_shopping"],
    "gastronomy": ["experiencia_patronicada", "reseña_comercial", "evento", "reservas"],
    "lifestyle": ["afiliación", "experiencia_de_marca", "membresía", "guía"],
    "luxury": ["lead_calificado", "evento_privado", "producción_premium", "alianza"],
    "monetization": ["lead_magnet", "afiliación", "servicio", "formación"],
}

RISKY_TERMS = [
    "guaranteed income", "get rich quick", "crypto pump", "casino", "betting",
    "ingreso garantizado", "dinero fácil", "apuesta", "rumor sin confirmar",
]


def classify_item(item: Dict[str, Any]) -> str:
    text = " ".join([
        str(item.get("title", "")),
        str(item.get("summary", "")),
        str(item.get("link", "")),
    ]).lower()

    for pillar, domains in DOMAIN_HINTS.items():
        if any(domain in text for domain in domains):
            return pillar

    scores = {
        pillar: sum(1 for keyword in words if keyword in f" {text} ")
        for pillar, words in KEYWORDS.items()
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "gaming"


def is_editorially_safe(item: Dict[str, Any], pillar: str) -> bool:
    text = " ".join([
        str(item.get("title", "")),
        str(item.get("summary", "")),
    ]).lower()
    return not any(term in text for term in RISKY_TERMS)


def commercial_angles_for_pillar(pillar: str) -> List[str]:
    return list(COMMERCIAL_ANGLES.get(pillar, []))


def rank_articles_by_strategy(
    articles: List[Dict[str, Any]],
    strategy: Dict[str, Any] | None = None,
    limit: int = 20,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    strategy = strategy or {}
    weights = strategy.get("pillar_weights") or DEFAULT_WEIGHTS
    rng = rng or random.Random()

    buckets: Dict[str, List[Dict[str, Any]]] = {pillar: [] for pillar in DEFAULT_WEIGHTS}
    for original in articles:
        item = dict(original)
        pillar = classify_item(item)
        if not is_editorially_safe(item, pillar):
            continue
        item["pillar"] = pillar
        item["commercial_angles"] = commercial_angles_for_pillar(pillar)
        buckets.setdefault(pillar, []).append(item)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    available = [pillar for pillar, bucket in buckets.items() if bucket]
    ranked: List[Dict[str, Any]] = []

    while available and len(ranked) < max(0, limit):
        chosen = rng.choices(
            available,
            weights=[max(0, float(weights.get(pillar, 0))) for pillar in available],
            k=1,
        )[0]
        ranked.append(buckets[chosen].pop())
        available = [pillar for pillar in available if buckets[pillar]]

    return ranked
