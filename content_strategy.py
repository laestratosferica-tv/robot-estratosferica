import random
from typing import Any, Dict, List


DEFAULT_WEIGHTS = {
    "gaming": 50,
    "technology": 25,
    "advertising": 15,
    "monetization": 10,
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
        "app", "creator tool", "virtual reality", "cybersecurity",
    ],
    "advertising": [
        "advertising", "marketing", "campaign", "brand", "branding", "creative",
        "social media", "content strategy", "audience", "creator economy",
    ],
    "monetization": [
        "monetization", "affiliate", "subscription", "sponsor", "revenue",
        "business model", "startup", "ecommerce", "product hunt", "creator fund",
    ],
}

DOMAIN_HINTS = {
    "gaming": ["dexerto.com", "pcgamer.com", "gamespot.com", "esports", "dotesports.com", "hltv.org"],
    "technology": ["techcrunch.com", "blog.google", "openai.com"],
    "advertising": ["hubspot.com", "marketingdirecto.com"],
    "monetization": ["producthunt.com"],
}

RISKY_MONETIZATION_TERMS = [
    "guaranteed income", "get rich quick", "crypto pump", "casino", "betting",
    "ingreso garantizado", "dinero fácil", "apuesta",
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
    if pillar != "monetization":
        return True
    text = " ".join([
        str(item.get("title", "")),
        str(item.get("summary", "")),
    ]).lower()
    return not any(term in text for term in RISKY_MONETIZATION_TERMS)


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
