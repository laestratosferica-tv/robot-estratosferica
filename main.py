import os
import json
import re
import time
import requests
import feedparser
import boto3
from datetime import datetime, timezone
from dateutil import parser as dtparser
from openai import OpenAI

# ============================================================
# LA ESTRATOSFÉRICA TV – ROBOT EDITORIAL ESPORTS
# Archivo único: main.py (REEMPLAZAR TODO por este contenido)
# ============================================================

# ==============================
# VARIABLES DE ENTORNO (Railway)
# ==============================
BUCKET_NAME = os.environ["BUCKET_NAME"]
R2_ENDPOINT = os.environ["R2_ENDPOINT_URL"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.4"))

MAX_HIGH = int(os.environ.get("MAX_HIGH", "3"))
MAX_MEDIUM = int(os.environ.get("MAX_MEDIUM", "5"))
MAX_LOW = int(os.environ.get("MAX_LOW", "4"))

# Para evitar gastar mucho al inicio:
MAX_ARTICLES_PER_RUN = int(os.environ.get("MAX_ARTICLES_PER_RUN", "8"))

# Si OpenAI falla, ¿seguimos con fallback?
ALLOW_FALLBACK_ON_OPENAI_ERROR = os.environ.get("ALLOW_FALLBACK_ON_OPENAI_ERROR", "true").lower() == "true"

# ==============================
# FUENTES RSS
# ==============================
RSS_FEEDS = [
    "https://www.dexerto.com/feed/",
    "https://esportsinsider.com/feed",
    "https://www.esports.net/news/feed/",
    "https://www.pcgamer.com/rss/",
]

# ==============================
# FILTROS DE CONTENIDO
# ==============================
KEYWORDS_INCLUDE = [
    "esports", "e-sports", "competitive", "competition", "pro scene", "tournament",
    "major", "worlds", "lan", "playoffs", "qualifier", "open qualifier", "split",

    "valorant", "vct",
    "cs2", "counter-strike", "cs:", "csgo",
    "call of duty", "cod", "warzone",
    "overwatch", "ow2",
    "apex", "apex legends",
    "rainbow six", "r6", "r6 siege",

    "league of legends", "lol", "lcs", "lec", "lck", "lpl", "msi",
    "dota", "dota 2", "ti", "the international",

    "street fighter", "sf6", "tekken", "mk1", "mortal kombat", "evo",
    "ea sports fc", "fc 24", "fc 25", "fc 26", "fifa",

    "esl", "blast", "pgl", "riot", "activision", "ubisoft",

    "roster", "fichaje", "plantilla", "transfer", "traspaso", "trade", "rumor", "signing",

    "f1 esports", "sim racing", "simracing", "gran turismo", "iracing",

    "ddr", "dance dance revolution", "just dance",

    "minecraft", "roblox", "roleplay", "rp",
    "gta", "gta v", "gta 6",
    "fortnite",

    "rocket league",
    "pubg",
    "mobile legends", "mlbb",
    "free fire",
]

KEYWORDS_EXCLUDE = [
    "nba", "nfl", "mlb", "nhl",
    "olympic", "olympics",
    "premier league", "la liga",
    "champions league",
]

HARD_PROMOTE_HIGH = [
    "récord", "record", "histórico", "historic",
    "sanción", "suspensión", "ban", "baneo",
    "trade", "traspaso", "fichaje",
    "polémica", "investigación",
    "campeón", "campeones", "champion",
    "comunicado oficial", "official statement",
    "descalificación", "disqualification",
    "cheating", "trampa", "hack", "hacks",
]

# ==============================
# CONEXIÓN R2
# ==============================
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)

# ==============================
# OPENAI CLIENT
# ==============================
client = OpenAI()

# ==============================
# HELPERS
# ==============================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def passes_filters(title: str) -> bool:
    t = normalize_text(title)
    if any(bad in t for bad in KEYWORDS_EXCLUDE):
        return False
    if not any(ok in t for ok in KEYWORDS_INCLUDE):
        return False
    return True

def try_get_excerpt(url: str, timeout: int = 8) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        html = r.text
        m = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.I
        )
        return m.group(1).strip() if m else ""
    except Exception:
        return ""

def save_to_r2(key: str, data) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=body,
        ContentType="application/json"
    )
    print("Archivo guardado en R2:", key)

# ==============================
# OPENAI: CLASIFICACIÓN + 6 POSTS THREADS
# ==============================
def fallback_editorial(article: dict, reason: str) -> dict:
    title = article.get("title", "")
    url = article.get("link", "")
    excerpt = article.get("excerpt", "")

    base = f"{title}. {excerpt}".strip()
    if len(base) > 220:
        base = base[:220].rstrip() + "..."

    posts = [
        f"{base} ¿Qué impacto crees que puede tener esto en la escena?",
        f"Contexto rápido: {base} ¿Te parece una decisión acertada?",
        f"Lectura editorial: {base} ¿Qué detalle te parece más importante?",
        f"Si esto se confirma, podría cambiar prioridades competitivas. ¿Estás de acuerdo?",
        f"Más allá del titular, el punto es el efecto en equipos/jugadores. ¿A quién afecta más?",
        f"Fuente: {url} ¿Qué te gustaría que analicemos a fondo de esta noticia?"
    ]

    return {
        "is_esports": True,
        "priority": "baja",
        "reason": reason,
        "threads_posts": posts,
        "topic_tags": ["gaming", "news", "LATAM"],
        "source_quality": "media"
    }

def openai_editorial(article: dict) -> tuple[dict, str]:
    """
    Retorna (editorial_json, error_code_str)
    error_code_str = "" si todo OK.
    """
    title = article.get("title", "")
    url = article.get("link", "")
    excerpt = article.get("excerpt", "")

    prompt = f"""
Eres el editor senior de LA ESTRATOSFÉRICA TV (esports LATAM).
Tarea: clasificar prioridad editorial y crear 6 posts para Threads.

Estilo obligatorio:
- Analítico, firme, profesional
- Sin exageraciones, sin clickbait
- Contexto + análisis + opinión con criterio
- Cierra CADA post con una pregunta inteligente
- Evita mayúsculas innecesarias y emojis excesivos (máximo 1 emoji por post, opcional)

Criterio de prioridad (impacto real):
- ALTA: cambios oficiales, sanciones, trades relevantes, resultados que cambian ranking/temporada, decisiones de liga, finales/majors/worlds, impacto LATAM claro o impacto global muy fuerte.
- MEDIA: noticias relevantes pero no determinantes (previas, declaraciones, rumores con respaldo, resultados menores).
- BAJA: notas livianas, curiosidades, contenido repetido, sin implicación competitiva.

Devuelve SOLO JSON válido con esta forma exacta:
{{
  "is_esports": true,
  "priority": "alta" | "media" | "baja",
  "reason": "string",
  "threads_posts": ["post1","post2","post3","post4","post5","post6"],
  "topic_tags": ["tag1","tag2","tag3"],
  "source_quality": "alta" | "media" | "baja"
}}

Noticia:
title: "{title}"
excerpt: "{excerpt}"
url: "{url}"
""".strip()

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Devuelve SOLO JSON válido. No incluyas texto fuera del JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=OPENAI_TEMPERATURE,
            response_format={"type": "json_object"},
        )

        text = resp.choices[0].message.content

        data = json.loads(text)

        # asegurar 6 posts
        posts = data.get("threads_posts", [])
        if not isinstance(posts, list):
            posts = []
        while len(posts) < 6:
            posts.append("")
        data["threads_posts"] = posts[:6]

        # asegurar 3 tags
        tags = data.get("topic_tags", [])
        if not isinstance(tags, list):
            tags = ["gaming", "news", "LATAM"]
        while len(tags) < 3:
            tags.append("LATAM")
        data["topic_tags"] = tags[:3]

        return data, ""

    except Exception as e:
        msg = str(e)
        # detecta cuota/429
        if "insufficient_quota" in msg or "Error code: 429" in msg:
            return {}, "insufficient_quota"
        # otros errores
        return {}, "openai_error"

# ==============================
# REGLAS DURAS (PROMOVER A ALTA)
# ==============================
def apply_hard_rules(article: dict) -> dict:
    t = normalize_text(article.get("title", "")) + " " + normalize_text(article.get("excerpt", ""))
    if any(w in t for w in HARD_PROMOTE_HIGH):
        article["editorial"]["priority"] = "alta"
        article["editorial"]["reason"] = (article["editorial"].get("reason", "") + " (Promovida por regla dura)").strip()
        article["promoted_by_rule"] = True
    else:
        article["promoted_by_rule"] = False
    return article

# ==============================
# CONTROL EDITORIAL (LIMITES)
# ==============================
def enforce_limits(enriched: list) -> tuple[list, dict]:
    enriched_sorted = sorted(enriched, key=lambda a: a.get("published", ""), reverse=True)

    highs, meds, lows = [], [], []
    for a in enriched_sorted:
        p = a["editorial"].get("priority", "baja")
        if p == "alta":
            highs.append(a)
        elif p == "media":
            meds.append(a)
        else:
            lows.append(a)

    if len(highs) > MAX_HIGH:
        overflow = highs[MAX_HIGH:]
        highs = highs[:MAX_HIGH]
        for a in overflow:
            a["editorial"]["priority"] = "media"
            meds.append(a)

    if len(meds) > MAX_MEDIUM:
        overflow = meds[MAX_MEDIUM:]
        meds = meds[:MAX_MEDIUM]
        for a in overflow:
            a["editorial"]["priority"] = "baja"
            lows.append(a)

    discarded = []
    if len(lows) > MAX_LOW:
        discarded = lows[MAX_LOW:]
        lows = lows[:MAX_LOW]

    final = highs + meds + lows

    metrics = {
        "final_high": len(highs),
        "final_medium": len(meds),
        "final_low": len(lows),
        "discarded_low": len(discarded),
        "promoted_by_hard_rules": sum(1 for a in enriched_sorted if a.get("promoted_by_rule")),
    }
    return final, metrics

# ==============================
# SCRAPER
# ==============================
def get_articles():
    articles = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            if not title:
                continue
            if not passes_filters(title):
                continue

            link = getattr(entry, "link", "")

            try:
                published_raw = getattr(entry, "published", "")
                published_dt = dtparser.parse(published_raw) if published_raw else datetime.now(timezone.utc)
            except Exception:
                published_dt = datetime.now(timezone.utc)

            excerpt = try_get_excerpt(link) if link else ""

            articles.append({
                "title": title,
                "link": link,
                "published": published_dt.isoformat(),
                "excerpt": excerpt,
                "source_feed": url
            })

    return articles

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("Obteniendo artículos (RSS + filtros)...")
    articles = get_articles()
    print(f"{len(articles)} artículos encontrados (filtrados)")

    # Limitar artículos por corrida para cuidar costo/cuota
    articles = articles[:MAX_ARTICLES_PER_RUN]
    print(f"Procesando con IA (máximo {MAX_ARTICLES_PER_RUN})...")

    enriched = []
    openai_errors = {"insufficient_quota": 0, "openai_error": 0}

    for a in articles:
        editorial, err = openai_editorial(a)

        if err:
            openai_errors[err] = openai_errors.get(err, 0) + 1
            if ALLOW_FALLBACK_ON_OPENAI_ERROR:
                a["editorial"] = fallback_editorial(a, reason=f"OpenAI error: {err}")
            else:
                # si no permites fallback, saltamos el item
                continue
        else:
            a["editorial"] = editorial

        a = apply_hard_rules(a)
        enriched.append(a)

        # pequeña pausa para no golpear rate limits
        time.sleep(0.2)

        # Si no hay cuota, no tiene sentido seguir llamando OpenAI
        if err == "insufficient_quota":
            break

    final, metrics = enforce_limits(enriched)

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "scraped_filtered": len(get_articles()),
            "processed_this_run": len(articles),
            "enriched": len(enriched),
            "openai_errors": openai_errors,
            **metrics
        },
        "items": final
    }

    key = f"editorial_run_{run_id}.json"
    save_to_r2(key, payload)
