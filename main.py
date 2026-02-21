print("RUNNING MULTIRED v1 (is_gaming + category)")

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

# ==============================
# FASTAPI (para OAuth Threads)
# ==============================
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

app = FastAPI()

# ==============================
# THREADS: CONFIG
# ==============================
THREADS_USER_ID = os.getenv("THREADS_USER_ID", "25864040949913281")
THREADS_STATE_KEY = os.getenv("THREADS_STATE_KEY", "threads_state.json")
THREADS_AUTO_POST = os.getenv("THREADS_AUTO_POST", "true").lower() == "true"
THREADS_AUTO_POST_LIMIT = int(os.getenv("THREADS_AUTO_POST_LIMIT", "1"))
THREADS_DRY_RUN = os.getenv("THREADS_DRY_RUN", "false").lower() == "true"

# ==============================
# THREADS: POST DE PRUEBA
# ==============================
@app.get("/post_test")
def post_test():
    """
    Publica un post de prueba en Threads usando el long-lived token guardado en Railway.
    Requiere variable:
      - THREADS_USER_ACCESS_TOKEN
    Opcional:
      - THREADS_USER_ID (si no existe, usa el default)
    """
    token = os.getenv("THREADS_USER_ACCESS_TOKEN")
    if not token:
        raise HTTPException(
            status_code=500,
            detail="Falta THREADS_USER_ACCESS_TOKEN en Railway (Variables).",
        )

    # 1) Crear contenedor del post (TEXT)
    create_url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    create_res = requests.post(
        create_url,
        data={
            "media_type": "TEXT",
            "text": "🚀 Prueba automática desde Robot Editorial La Estratosférica TV",
            "access_token": token,
        },
        timeout=30,
    )
    try:
        create_data = create_res.json()
    except Exception:
        create_data = {"raw": create_res.text}

    if create_res.status_code != 200 or "id" not in create_data:
        return {
            "step": "create_container_failed",
            "status": create_res.status_code,
            "response": create_data,
        }

    creation_id = create_data["id"]

    # 2) Publicar el contenedor
    publish_url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    publish_res = requests.post(
        publish_url,
        data={"creation_id": creation_id, "access_token": token},
        timeout=30,
    )
    try:
        publish_data = publish_res.json()
    except Exception:
        publish_data = {"raw": publish_res.text}

    return {"step": "ok", "container": create_data, "publish": publish_data}


@app.get("/health")
def health():
    return {"ok": True, "service": "robot-estratosferica"}


# ==============================
# INICIO OAUTH (Threads)
# ==============================
@app.get("/login")
def threads_login():
    """
    Inicia el OAuth de Threads. Abre:
    /login  -> redirige a threads.net/oauth/authorize
    """
    THREADS_APP_ID = os.getenv("THREADS_APP_ID")
    THREADS_REDIRECT_URI = os.getenv("THREADS_REDIRECT_URI")

    missing = []
    if not THREADS_APP_ID:
        missing.append("THREADS_APP_ID")
    if not THREADS_REDIRECT_URI:
        missing.append("THREADS_REDIRECT_URI")

    if missing:
        return JSONResponse(
            {"error": "Faltan variables en Railway", "missing": missing},
            status_code=400,
        )

    auth_url = (
        "https://www.threads.net/oauth/authorize"
        f"?client_id={THREADS_APP_ID}"
        f"&redirect_uri={THREADS_REDIRECT_URI}"
        "&response_type=code"
        "&scope=threads_basic"
    )
    return RedirectResponse(auth_url)


@app.get("/callback/")
def threads_callback(request: Request):
    """
    Threads redirige aquí con: /callback/?code=...
    Este endpoint cambia code -> short-lived token -> long-lived token
    """
    code = request.query_params.get("code")
    if not code:
        return JSONResponse(
            {"error": "No llegó 'code'. Debe ser /callback/?code=..."},
            status_code=400,
        )

    THREADS_APP_ID = os.getenv("THREADS_APP_ID")
    THREADS_APP_SECRET = os.getenv("THREADS_APP_SECRET")
    THREADS_REDIRECT_URI = os.getenv("THREADS_REDIRECT_URI")

    missing = []
    if not THREADS_APP_ID:
        missing.append("THREADS_APP_ID")
    if not THREADS_APP_SECRET:
        missing.append("THREADS_APP_SECRET")
    if not THREADS_REDIRECT_URI:
        missing.append("THREADS_REDIRECT_URI")

    if missing:
        return JSONResponse(
            {"error": "Faltan variables en Railway", "missing": missing},
            status_code=400,
        )

    # 1) code -> short-lived token
    token_url = "https://graph.threads.net/oauth/access_token"
    data = {
        "client_id": THREADS_APP_ID,
        "client_secret": THREADS_APP_SECRET,
        "grant_type": "authorization_code",
        "redirect_uri": THREADS_REDIRECT_URI,
        "code": code,
    }

    r = requests.post(token_url, data=data, timeout=30)
    try:
        short_payload = r.json()
    except Exception:
        short_payload = {"raw": r.text}

    if r.status_code != 200 or "access_token" not in short_payload:
        return JSONResponse(
            {
                "step": "code_to_token_failed",
                "status": r.status_code,
                "response": short_payload,
            },
            status_code=400,
        )

    short_token = short_payload["access_token"]

    # 2) short-lived -> long-lived (60 días)
    ll_url = "https://graph.threads.net/access_token"
    params = {
        "grant_type": "th_exchange_token",
        "client_secret": THREADS_APP_SECRET,
        "access_token": short_token,
    }

    r2 = requests.get(ll_url, params=params, timeout=30)
    try:
        long_payload = r2.json()
    except Exception:
        long_payload = {"raw": r2.text}

    if r2.status_code != 200 or "access_token" not in long_payload:
        return JSONResponse(
            {
                "ok": True,
                "short_lived": short_payload,
                "long_lived_error": {"status": r2.status_code, "response": long_payload},
                "next": "Copia short_lived.access_token (dura ~1h) o revisa permisos/config.",
            }
        )

    return JSONResponse(
        {
            "ok": True,
            "long_lived": long_payload,
            "short_lived": short_payload,
            "next": "Copia long_lived.access_token y guárdalo en Railway como THREADS_USER_ACCESS_TOKEN.",
        }
    )


# ============================================================
# ROBOT EDITORIAL
# ============================================================

# ==============================
# VARIABLES DE ENTORNO (Railway/GitHub Actions)
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

ALLOW_FALLBACK_ON_OPENAI_ERROR = (
    os.environ.get("ALLOW_FALLBACK_ON_OPENAI_ERROR", "true").lower() == "true"
)

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
KEYWORDS_INCLUDE = [
    "gaming", "video game", "videogame", "game", "gamer", "juego", "juegos",
    "playstation", "ps5", "ps4", "xbox", "nintendo", "switch", "steam", "pc",
    "ubisoft", "riot", "blizzard", "ea", "epic", "bandai", "capcom", "square enix", "take-two",
    "esports", "e-sports", "competitive", "tournament", "major", "worlds", "lan",
    "valorant", "vct",
    "cs2", "counter-strike",
    "league of legends", "lol", "lck", "lec", "lcs", "lpl", "msi",
    "dota", "dota 2", "the international",
    "fortnite", "call of duty", "cod", "warzone",
    "overwatch", "apex",
    "free fire", "mlbb", "mobile legends",
    "minecraft", "roblox",
    "just dance",
    "mario kart", "mariokart",
    "ea sports fc", "fc 24", "fc 25", "fc 26", "fifa",
    "rocket league", "pubg",
    "gta", "gta v", "gta 6",
    "pokemon", "zelda", "mario", "smash", "smash bros",
    "street fighter", "tekken", "mortal kombat",
]

KEYWORDS_EXCLUDE = [
    "nba", "nfl", "mlb", "nhl",
    "premier league", "champions league", "la liga",
    "olympic", "olympics",
]

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

def passes_filters(title: str, link: str, excerpt: str) -> bool:
    t = normalize_text(title)
    e = normalize_text(excerpt)
    u = (link or "").lower()

    if url_is_bad(u):
        return False

    if any(bad in t for bad in KEYWORDS_EXCLUDE):
        return False

    hay_keyword = (
        any(ok in t for ok in KEYWORDS_INCLUDE)
        or any(ok in e for ok in KEYWORDS_INCLUDE)
        or any(ok in u for ok in KEYWORDS_INCLUDE)
    )
    if not hay_keyword:
        return False

    return True

def save_to_r2(key: str, data) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=body,
        ContentType="application/json"
    )
    print("Archivo guardado en R2:", key)

def load_from_r2(key: str):
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None

# ==============================
# THREADS: STATE (para no repetir links)
# ==============================
def load_threads_state() -> dict:
    state = load_from_r2(THREADS_STATE_KEY)
    if not isinstance(state, dict):
        state = {}
    posted = state.get("posted_links", [])
    if not isinstance(posted, list):
        posted = []
    # mantén máximo 200 para no crecer infinito
    state["posted_links"] = posted[-200:]
    return state

def save_threads_state(state: dict) -> None:
    save_to_r2(THREADS_STATE_KEY, state)

# ==============================
# THREADS: PUBLICAR
# ==============================
def threads_publish_text(text: str) -> dict:
    token = os.getenv("THREADS_USER_ACCESS_TOKEN")
    if not token:
        return {"ok": False, "error": "missing_THREADS_USER_ACCESS_TOKEN"}

    # Protección: Threads tiene límite de texto (varía), recortamos conservador
    text = (text or "").strip()
    if len(text) > 480:
        text = text[:477].rstrip() + "..."

    if THREADS_DRY_RUN:
        return {"ok": True, "dry_run": True, "text": text}

    # 1) Crear contenedor
    create_url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    create_res = requests.post(
        create_url,
        data={
            "media_type": "TEXT",
            "text": text,
            "access_token": token,
        },
        timeout=30,
    )
    try:
        create_data = create_res.json()
    except Exception:
        create_data = {"raw": create_res.text}

    if create_res.status_code != 200 or "id" not in create_data:
        return {
            "ok": False,
            "step": "create_container_failed",
            "status": create_res.status_code,
            "response": create_data,
        }

    creation_id = create_data["id"]

    # 2) Publicar
    publish_url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    publish_res = requests.post(
        publish_url,
        data={"creation_id": creation_id, "access_token": token},
        timeout=30,
    )
    try:
        publish_data = publish_res.json()
    except Exception:
        publish_data = {"raw": publish_res.text}

    return {
        "ok": True,
        "container": create_data,
        "publish": publish_data,
        "text": text,
    }

def build_threads_text(item: dict) -> str:
    """
    Toma 1 item final y genera el texto a publicar:
    - usa threads_posts[0] si existe
    - agrega la fuente (link)
    """
    title = (item.get("title") or "").strip()
    link = (item.get("link") or "").strip()
    editorial = item.get("editorial") or {}
    posts = editorial.get("threads_posts") or []
    first = ""
    if isinstance(posts, list) and len(posts) > 0:
        first = (posts[0] or "").strip()

    if not first:
        first = title

    if link:
        # agrega fuente al final
        text = f"{first}\n\nFuente: {link}".strip()
    else:
        text = first

    return text

def pick_best_item_to_post(final_items: list, posted_links: set):
    """
    Elige el mejor item:
    - prioridad alta > media > baja
    - más reciente
    - que NO esté ya publicado
    """
    def priority_rank(p: str) -> int:
        if p == "alta":
            return 3
        if p == "media":
            return 2
        return 1

    candidates = []
    for it in final_items:
        link = (it.get("link") or "").strip()
        if link and link in posted_links:
            continue
        editorial = it.get("editorial") or {}
        p = editorial.get("priority", "baja")
        pub = it.get("published", "")
        candidates.append((priority_rank(p), pub, it))

    # orden: rank desc, published desc
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2] if candidates else None

# ==============================
# FALLBACK EDITORIAL
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
        "Por qué importa: señal de comunidad/mercado.",
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
# OPENAI: MULTIPLATAFORMA
# ==============================
def openai_editorial(article: dict):
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
2) Categoría: "competitive" (competitivo/esports), "industry" (negocio/mercado), "culture" (cultura gamer), "other" (no gaming).
3) Prioridad: "alta", "media", "baja".
4) Genera piezas MULTIPLATAFORMA:

THREADS:
- 6 posts estilo briefing ejecutivo.

INSTAGRAM:
- Carrusel 6 slides.
- 1 caption (100–180 palabras).

TIKTOK/SHORTS:
- Guion 35–55 segundos.

YOUTUBE:
- Outline 6–8 bullets.

FACEBOOK:
- 1 post 80–160 palabras.

Devuelve SOLO JSON válido EXACTO con las llaves definidas.

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

        posts = data.get("threads_posts", [])
        if not isinstance(posts, list):
            posts = []
        while len(posts) < 6:
            posts.append("")
        data["threads_posts"] = posts[:6]

        slides = data.get("instagram_carousel_slides", [])
        if not isinstance(slides, list):
            slides = []
        while len(slides) < 6:
            slides.append("")
        data["instagram_carousel_slides"] = slides[:6]

        outline = data.get("youtube_outline", [])
        if not isinstance(outline, list):
            outline = []
        while len(outline) < 6:
            outline.append("")
        data["youtube_outline"] = outline[:8]

        tags = data.get("topic_tags", [])
        if not isinstance(tags, list):
            tags = ["gaming", "ecosistema", "LATAM"]
        while len(tags) < 3:
            tags.append("LATAM")
        data["topic_tags"] = tags[:3]

        if not isinstance(data.get("instagram_caption", ""), str):
            data["instagram_caption"] = ""
        if not isinstance(data.get("tiktok_script", ""), str):
            data["tiktok_script"] = ""
        if not isinstance(data.get("facebook_post", ""), str):
            data["facebook_post"] = ""

        if "is_gaming" not in data:
            data["is_gaming"] = True
        if "category" not in data:
            data["category"] = "culture"
        if "priority" not in data:
            data["priority"] = "baja"
        if "reason" not in data:
            data["reason"] = "Sin razón proporcionada por el modelo."

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
def enforce_limits(items: list):
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
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            title = getattr(entry, "title", "")
            if not title:
                continue

            link = getattr(entry, "link", "")
            if link and url_is_bad(link):
                continue

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
                "source_feed": feed_url
            })

    return articles

# ==============================
# EJECUCIÓN DEL ROBOT
# ==============================
def run_robot_once():
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

        if not a["editorial"].get("is_gaming", True) or a["editorial"].get("category") == "other":
            discarded_non_gaming += 1
            continue

        enriched.append(a)
        time.sleep(0.25)

        if err == "insufficient_quota":
            break

    final, metrics = enforce_limits(enriched)
    print("FINAL ITEMS:", len(final))
    
    # ==============================
    # AUTO-POST (Threads): 1 post por corrida
    # ==============================
    threads_publish_result = None
    if THREADS_AUTO_POST and final and THREADS_AUTO_POST_LIMIT > 0:
        state = load_threads_state()
        posted_links = set(state.get("posted_links", []))

        item = pick_best_item_to_post(final, posted_links)
        if item:
            text = build_threads_text(item)
            print("Auto-post Threads: publicando 1 post...")
            res = threads_publish_text(text)
            threads_publish_result = res

            # marca como publicado si salió ok (y tiene link)
            if res.get("ok") and (item.get("link") or "").strip():
                posted_links.add(item["link"].strip())
                state["posted_links"] = list(posted_links)[-200:]
                state["last_posted_at"] = datetime.now(timezone.utc).isoformat()
                state["last_posted_link"] = item["link"].strip()
                save_threads_state(state)
                print("Auto-post Threads: OK ✅")
            else:
                print("Auto-post Threads: NO OK ⚠️", res)
        else:
            print("Auto-post Threads: no hay item nuevo (evitando repetidos).")

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
        "threads_auto_post": {
            "enabled": THREADS_AUTO_POST,
            "limit": THREADS_AUTO_POST_LIMIT,
            "dry_run": THREADS_DRY_RUN,
            "result": threads_publish_result,
        },
        "items": final
    }

    key = f"editorial_run_{run_id}.json"
    save_to_r2(key, payload)

if __name__ == "__main__":
    # Esto SOLO se ejecuta si corres: python main.py
    # NO se ejecuta cuando Railway usa uvicorn.
    run_robot_once()
