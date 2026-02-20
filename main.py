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
# LA ESTRATOSFÉRICA TV – ROBOT EDITORIAL ASPIRACIONAL
# ============================================================

# ==============================
# VARIABLES DE ENTORNO
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
MAX_ARTICLES_PER_RUN = int(os.environ.get("MAX_ARTICLES_PER_RUN", "5"))

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
# FILTRO BÁSICO
# ==============================
KEYWORDS_INCLUDE = [
    "esports", "gaming", "valorant", "cs2", "counter-strike",
    "league of legends", "lol", "dota", "fortnite",
    "call of duty", "overwatch", "apex",
    "roblox", "minecraft", "gta",
    "playstation", "xbox", "nintendo",
    "tournament", "major", "worlds", "lan",
]

KEYWORDS_EXCLUDE = [
    "nba", "nfl", "mlb", "nhl",
    "premier league", "champions league"
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

def try_get_excerpt(url: str, timeout: int = 6) -> str:
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
    except:
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
# OPENAI EDITORIAL
# ==============================
def openai_editorial(article: dict):
    title = article.get("title", "")
    url = article.get("link", "")
    excerpt = article.get("excerpt", "")

    prompt = f"""
Eres el editor ejecutivo de LA ESTRATOSFÉRICA TV.

Medio aspiracional.
Habla al jugador, creador, empresario y sponsor.
Profesional, claro y estratégico.
Sin exageraciones.
Sin clickbait.
Sin tono infantil.

Tarea:
1) Determinar si pertenece al ecosistema gaming/esports.
2) Clasificar prioridad.
3) Generar 6 posts estilo briefing ejecutivo.

Cada post debe:
- Explicar por qué importa
- Dar lectura estratégica
- Cerrar con una pregunta profesional

Clasificación:
ALTA = impacto competitivo real
MEDIA = movimiento de industria o tendencia relevante
BAJA = cultura gamer o marketing

Devuelve SOLO JSON válido:

{{
  "is_esports": true,
  "priority": "alta" | "media" | "baja",
  "reason": "explicación ejecutiva",
  "threads_posts": ["p1","p2","p3","p4","p5","p6"],
  "topic_tags": ["t1","t2","t3"],
  "source_quality": "alta" | "media" | "baja"
}}

Noticia:
title: "{title}"
excerpt: "{excerpt}"
url: "{url}"
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Responde SOLO JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=OPENAI_TEMPERATURE,
            response_format={"type": "json_object"},
        )

        return json.loads(resp.choices[0].message.content)

    except Exception as e:
        print("OpenAI error:", e)
        return None

# ==============================
# CONTROL EDITORIAL
# ==============================
def enforce_limits(enriched):
    highs = [a for a in enriched if a["editorial"]["priority"] == "alta"]
    meds = [a for a in enriched if a["editorial"]["priority"] == "media"]
    lows = [a for a in enriched if a["editorial"]["priority"] == "baja"]

    highs = highs[:MAX_HIGH]
    meds = meds[:MAX_MEDIUM]
    lows = lows[:MAX_LOW]

    return highs + meds + lows

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
            except:
                published_dt = datetime.now(timezone.utc)

            excerpt = try_get_excerpt(link)

            articles.append({
                "title": title,
                "link": link,
                "published": published_dt.isoformat(),
                "excerpt": excerpt
            })

    return articles

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("Obteniendo artículos...")
    articles = get_articles()
    print(f"{len(articles)} encontrados")

    articles = articles[:MAX_ARTICLES_PER_RUN]
    print(f"Procesando máximo {MAX_ARTICLES_PER_RUN}")

    enriched = []

    for a in articles:
        editorial = openai_editorial(a)
        if editorial is None:
            continue

        if not editorial.get("is_esports", True):
            continue

        a["editorial"] = editorial
        enriched.append(a)

        time.sleep(0.3)

    final = enforce_limits(enriched)

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(final),
        "items": final
    }

    save_to_r2(f"editorial_run_{run_id}.json", payload)
