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
# LA ESTRATOSFÉRICA TV – ROBOT EDITORIAL (ECOSISTEMA GAMER AMPLIO + MULTIRED)
# Archivo único: main.py (REEMPLAZAR TODO)
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

# Para empezar barato:
MAX_ARTICLES_PER_RUN = int(os.environ.get("MAX_ARTICLES_PER_RUN", "5"))

# Si OpenAI falla, igual guardamos (fallback):
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
# FILTRO DE ENTRADA (AMPLIO)
# ==============================
# Este es un filtro inicial para evitar ruido.
# La IA hace la decisión final con is_gaming + category.
KEYWORDS_INCLUDE = [
    # general gaming
    "gaming", "video game", "videogame", "game", "gamer", "juego", "juegos",
    "playstation", "ps5", "ps4", "xbox", "nintendo", "switch", "steam", "pc",

    # publishers / estudios
    "ubisoft", "riot", "blizzard", "ea", "epic", "bandai", "capcom", "square enix", "take-two",

    # esports / competitivo
    "esports", "e-sports", "competitive", "tournament", "major", "worlds", "lan",
    "valorant", "vct",
    "cs2", "counter-strike",
    "league of legends", "lol", "lck", "lec", "lcs", "lpl", "msi",
    "dota", "dota 2", "the international",
    "fortnite", "call of duty", "cod", "warzone",
    "overwatch", "apex",

    # mobile / latam
    "free fire", "mlbb", "mobile legends",

    # cultura gamer (lo que pediste)
    "minecraft", "roblox",
    "just dance",
    "mario kart", "mariokart",
    "ea sports fc", "fc 24", "fc 25", "fc 26", "fifa",
    "rocket league", "pubg",
    "gta", "gta v", "gta 6",
    "pokemon", "zelda", "mario", "smash", "smash bros",
    "street fighter", "tekken", "mortal kombat",
]

# deportes tradicionales
KEYWORDS_EXCLUDE = [
    "nba", "nfl", "mlb", "nhl",
    "premier league", "champions league", "la liga",
    "olympic", "olympics",
]

# filtro por URL para evitar secciones no gaming (muy efectivo)
URL_PATH_EXCLUDE = [
    "/tiktok/",
    "/tv-movies/",
    "/movies/",
    "/celebrity/",
    "/dating/",
]

# ==============================
# CONEXIÓN R2 (Cloudflare R2 via S3)
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

def url_is_bad(url: str) -> bool:
    u = (url or "").lower()
    return any(p in u for p in URL_PATH_EXCLUDE)

def passes_filters(title: str, link: str, excerpt: str) -> bool:
    t = normalize_text(title)
    e = normalize_text(excerpt)
    u = (link or "").lower()

    if url_is_bad(u):
        return False

    if any(bad in t for bad in KEYWORDS_EXCLUDE):
        return False

    # incluir si aparece keyword en title o excerpt o url
    hay_keyword = any(ok in t for ok in KEYWORDS_INCLUDE) or any(ok in e for ok in KEYWORDS_INCLUDE) or any(ok in u for ok in KEYWORDS_INCLUDE)
    if not hay_keyword:
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
# FALLBACK EDITORIAL (si OpenAI falla)
# ==============================
def fallback_editorial(article: dict, reason: str) -> dict:
    title = (article.get("title", "") or "").strip()
    url = (article.get("link", "") or "").strip()
    excerpt = (article.get("excerpt", "") or "").strip()

    base = f"{title}. {excerpt}".strip()
    if len(base) > 220:
        base = base[:220].rstrip() + "..."

    threads = [
        f"{base} ¿Qué lectura harías si esto fuera tu producto o comunidad?",
        "Señal clave: atención vs. retención. ¿Qué variable crees que manda aquí?",
        "Si eres jugador/creador: ¿cómo te afecta? Si eres marca: ¿qué oportunidad ves?",
        "En el ecosistema gamer, lo sostenible es ejecución. ¿Qué harías esta semana para sostenerlo?",
        "Pregunta de negocio: ¿esto construye valor a largo plazo o solo un pico de conversación?",
        f"Fuente: {url} ¿Qué ángulo quieres que profundicemos con datos?"
    ]

    ig_slides = [
        f"Titular: {title}",
        f"Contexto: {(excerpt[:140] + '...') if len(excerpt) > 140 else excerpt}",
        "Por qué importa: señal de mercado/comunidad.",
        "Lectura: producto + distribución + retención.",
        "Decisión: ¿qué priorizas hoy?",
        f"Fuente: {url}"
    ]

    return {
        "is_gaming": True,
        "category": "culture",
        "priority": "baja",
        "reason": reason,
        "threads_posts": threads,
        "instagram_carousel_slides": ig_slides,
        "instagram_caption": f"{title}\n\nLectura: esto es más señal de mercado/comunidad que un titular competitivo.\n\nSi tú lideraras el proyecto, ¿qué priorizarías esta semana?\n\nFuente: {url}",
        "tiktok_script": f"Brief gamer: {title}. Contexto: {excerpt}. Lectura: aquí manda la retención y el valor de comunidad. Pregunta: si fueras responsable, ¿qué decisión tomarías hoy?",
        "youtube_outline": [
            "Resumen en 30 segundos",
            "Contexto: por qué esto aparece ahora",
            "Qué significa para jugadores/creadores/marcas",
            "Riesgos y oportunidades",
            "Qué métricas observar",
            "Cierre: decisión de la semana"
        ],
        "facebook_post": f"{title}\n\nContexto: {excerpt}\n\nLectura: ¿qué pesa más aquí: producto, contenido o comunidad?\n\nFuente: {url}",
        "topic_tags": ["gaming", "ecosistema", "LATAM"],
        "source_quality": "media"
    }

# ==============================
# OPENAI: MULTIPLATAFORMA EN 1 SOLA LLAMADA
# ==============================
def openai_editorial(article: dict) -> tuple[dict, str]:
    title = article.get("title", "")
    url = article.get("link", "")
    excerpt = article.get("excerpt", "")

    prompt = f"""
Eres el editor ejecutivo de LA ESTRATOSFÉRICA TV.

Identidad del medio:
- Aspiracional: eleva al lector (jugador, creador, empresario, sponsor)
- Profesional, directo, claro
- Inclusivo: explica sin descalificar
- Sin clickbait, sin exageraciones, sin tono infantil

Tareas:
1) Determina si pertenece al ecosistema GAMING (no solo esports).
2) Clasifica categoría: "competitive" (competitivo/esports), "industry" (negocio/mercado), "culture" (cultura gamer), "other" (no gaming).
3) Prioridad editorial: "alta", "media", "baja".
4) Genera piezas MULTIPLATAFORMA:

THREADS:
- 6 posts, briefing ejecutivo
- Cada post: por qué importa + lectura estratégica + pregunta final profesional
- Evita repetir “¿qué opinas?” (usa decisión/riesgo/oportunidad/aprendizaje)

INSTAGRAM:
- Carrusel de 6 slides (texto corto por slide)
- 1 caption (100–180 palabras, tono profesional)

TIKTOK/SHORTS:
- Guion 35–55 segundos (voz), gancho sobrio + contexto + lectura + pregunta final

YOUTUBE:
- Outline para análisis semanal (6–8 bullets), "cómo funciona el mundo profesional" del gaming/esports

FACEBOOK:
- 1 post 80–160 palabras, comunidad + pregunta final

Reglas:
- Si NO es gaming: is_gaming=false, category="other", priority="baja"
- Aun si es "other", llena campos con texto mínimo (para logging), sin inventar hechos.

Devuelve SOLO JSON válido EXACTO con esta forma:

{{
  "is_gaming": true,
  "category": "competitive" | "industry" | "culture" | "other",
  "priority": "alta" | "media" | "baja",
  "reason": "explicación ejecutiva y breve",
  "threads_posts": ["p1","p2","p3","p4","p5","p6"],
  "instagram_carousel_slides": ["s1","s2","s3","s4","s5","s6"],
  "instagram_caption": "string",
  "tiktok_script": "string",
  "youtube_outline": ["b1","b2","b3","b4","b5","b6"],
  "facebook_post": "string",
  "topic_tags": ["t1","t2","t3"],
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
                {"role": "system", "content": "Responde SOLO JSON válido."},
                {"role": "user", "content": prompt},
            ],
            temperature=OPENAI_TEMPERATURE,
            response_format={"type": "json_object"},
        )

        text = resp.choices[0].message.content
        data = json.loads(text)

        # asegurar 6 threads
        posts = data.get("threads_posts", [])
        if not isinstance(posts, list):
            posts = []
        while len(posts) < 6:
            posts.append("")
        data["threads_posts"] = posts[:6]

        # asegurar 6 slides IG
        slides = data.get("instagram_carousel_slides", [])
        if not isinstance(slides, list):
            slides = []
        while len(slides) < 6:
            slides.append("")
        data["instagram_carousel_slides"] = slides[:6]

        # asegurar outline youtube mínimo 6
        outline = data.get("youtube_outline", [])
        if not isinstance(outline, list):
            outline = []
        while len(outline) < 6:
            outline.append("")
        data["youtube_outline"] = outline[:8]

        # asegurar 3 tags
        tags = data.get("topic_tags", [])
        if not isinstance(tags, list):
            tags = ["gaming", "ecosistema", "LATAM"]
        while len(tags) < 3:
            tags.append("LATAM")
        data["topic_tags"] = tags[:3]

        # asegurar strings
        if not isinstance(data.get("instagram_caption", ""), str):
            data["instagram_caption"] = ""
        if not isinstance(data.get("tiktok_script", ""), str):
            data["tiktok_script"] = ""
        if not isinstance(data.get("facebook_post", ""), str):
            data["facebook_post"] = ""

        # si el modelo devuelve vacíos, rellenar suavemente (no publicable pero no rompe)
        if all((p or "").strip() == "" for p in data["threads_posts"]):
            fb = fallback_editorial(article, reason="Relleno automático por posts vacíos del modelo.")
            data["threads_posts"] = fb["threads_posts"]

        return data, ""

    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "Error code: 429" in msg:
            return {}, "insufficient_quota"
        return {}, "openai_error"

# ==============================
# CONTROL EDITORIAL (LIMITES)
# ==============================
def enforce_limits(items: list) -> tuple[list, dict]:
    # Orden por fecha (más reciente primero)
    items_sorted = sorted(items, key=lambda a: a.get("published", ""), reverse=True)

    highs, meds, lows = [], [], []
    for a in items_sorted:
        p = a["editorial"].get("priority", "baja")
        if p == "alta":
            highs.append(a)
        elif p == "media":
            meds.append(a)
        else:
            lows.append(a)

    highs = highs[:MAX_HIGH]
    meds = meds[:MAX_MEDIUM]
    lows = lows[:MAX_LOW]

    final = highs + meds + lows

    metrics = {
        "final_high": len(highs),
        "final_medium": len(meds),
        "final_low": len(lows),
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

            link = getattr(entry, "link", "")
            excerpt = try_get_excerpt(link) if link else ""

            if not passes_filters(title, link, excerpt):
                continue

            try:
                published_raw = getattr(entry, "published", "")
                published_dt = dtparser.parse(published_raw) if published_raw else datetime.now(timezone.utc)
            except Exception:
                published_dt = datetime.now(timezone.utc)

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
    scraped = get_articles()
    print(f"{len(scraped)} artículos encontrados (filtrados)")

    articles = scraped[:MAX_ARTICLES_PER_RUN]
    print(f"Procesando con IA (máximo {MAX_ARTICLES_PER_RUN})...")

    enriched = []
    openai_errors = {"insufficient_quota": 0, "openai_error": 0}
    discarded_non_gaming = 0

    for a in articles:
        editorial, err = openai_editorial(a)

        if err:
            openai_errors[err] = openai_errors.get(err, 0) + 1
            if ALLOW_FALLBACK_ON_OPENAI_ERROR:
                a["editorial"] = fallback_editorial(a, reason=f"OpenAI error: {err}")
            else:
                continue
        else:
            a["editorial"] = editorial

        # DESCARTAR si NO es gaming (aquí se va TikTok/TV/fanfic)
        if not a["editorial"].get("is_gaming", True) or a["editorial"].get("category") == "other":
            discarded_non_gaming += 1
            continue

        enriched.append(a)

        time.sleep(0.25)

        # si se quedó sin cuota, no seguir llamando
        if err == "insufficient_quota":
            break

    final, metrics = enforce_limits(enriched)

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "scraped_filtered": len(scraped),
            "processed_this_run": len(articles),
            "enriched_after_ai": len(enriched),
            "discarded_non_gaming": discarded_non_gaming,
            "openai_errors": openai_errors,
            **metrics
        },
        "items": final
    }

    key = f"editorial_run_{run_id}.json"
    save_to_r2(key, payload)
