# main.py
# Robot Editorial + Auto-post Threads con re-host de imágenes en R2
# PROD v7: auto-mix feeds + balance por feed + Threads WAIT + anti-duplicado + repost
# + IG Queue (reels>carousel>image) guardado en R2: ugc/ig_queue/

import os
import re
import json
import time
import hashlib
import random
from datetime import datetime, timezone
from urllib.parse import urljoin
from typing import Optional, Tuple, Dict, Any, List

import requests
import boto3

# RSS
try:
    import feedparser
except Exception:
    feedparser = None

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


print("RUNNING MULTIRED v7 (PROD: MIX+BALANCE+STATE+THREADS WAIT + IG QUEUE)")

# =========================
# CONFIG
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

R2_PUBLIC_BASE_URL = os.getenv(
    "R2_PUBLIC_BASE_URL",
    "https://pub-8937244ee725495691514507bb8f431e.r2.dev"
).rstrip("/")

THREADS_USER_ID = os.getenv("THREADS_USER_ID", "me")  # recomendado: "me"
THREADS_USER_ACCESS_TOKEN = os.getenv("THREADS_USER_ACCESS_TOKEN")

THREADS_STATE_KEY = os.getenv("THREADS_STATE_KEY", "threads_state.json")

THREADS_AUTO_POST = os.getenv("THREADS_AUTO_POST", "true").lower() == "true"
THREADS_AUTO_POST_LIMIT = int(os.getenv("THREADS_AUTO_POST_LIMIT", "1"))
THREADS_DRY_RUN = os.getenv("THREADS_DRY_RUN", "false").lower() == "true"

# Repost logic
REPOST_MAX_TIMES = int(os.getenv("REPOST_MAX_TIMES", "3"))        # n=3
REPOST_WINDOW_DAYS = int(os.getenv("REPOST_WINDOW_DAYS", "7"))    # ventana=7
REPOST_ENABLE = os.getenv("REPOST_ENABLE", "true").lower() == "true"

# RSS feeds (coma)
RSS_FEEDS = [x.strip() for x in (os.getenv("RSS_FEEDS") or "").split(",") if x.strip()]
if not RSS_FEEDS:
    # Si no configuras RSS_FEEDS, será Dexerto siempre.
    RSS_FEEDS = ["https://www.dexerto.com/feed/"]

# Auto-mix / balance
MAX_PER_FEED = int(os.getenv("MAX_PER_FEED", "3"))  # default 3 (más diversidad)
SHUFFLE_ARTICLES = os.getenv("SHUFFLE_ARTICLES", "true").lower() == "true"

# Pool de candidatos a considerar por corrida
MAX_AI_ITEMS = int(os.getenv("MAX_AI_ITEMS", "15"))  # default 15 (más variedad)
SLEEP_BETWEEN_ITEMS_SEC = float(os.getenv("SLEEP_BETWEEN_ITEMS_SEC", "0.05"))

# Threads host correcto (SIN /v20.0)
THREADS_GRAPH = os.getenv("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")

# HTTP tuning
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
POST_RETRY_MAX = int(os.getenv("POST_RETRY_MAX", "2"))
POST_RETRY_SLEEP = float(os.getenv("POST_RETRY_SLEEP", "2"))

CONTAINER_WAIT_TIMEOUT = int(os.getenv("CONTAINER_WAIT_TIMEOUT", "120"))
CONTAINER_POLL_INTERVAL = float(os.getenv("CONTAINER_POLL_INTERVAL", "2"))

# IG Queue
IG_QUEUE_PREFIX = os.getenv("IG_QUEUE_PREFIX", "ugc/ig_queue").strip().strip("/")
IG_CAROUSEL_MAX_IMAGES = int(os.getenv("IG_CAROUSEL_MAX_IMAGES", "5"))

# =========================
# TIME
# =========================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_now() -> str:
    return now_utc().isoformat()

def parse_iso(dt: str) -> datetime:
    return datetime.fromisoformat(dt.replace("Z", "+00:00"))

def days_since(dt_iso: str) -> int:
    try:
        dt = parse_iso(dt_iso)
        return int((now_utc() - dt).total_seconds() // 86400)
    except Exception:
        return 999999

# =========================
# DEBUG / HTTP
# =========================

def _threads_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def _raise_meta_error(r: requests.Response, label: str = "META") -> None:
    if r.status_code < 400:
        return

    print(f"\n====== {label} ERROR DEBUG ======")
    print("URL:", r.request.url)
    print("METHOD:", r.request.method)
    print("STATUS:", r.status_code)
    if r.request.body:
        body = r.request.body
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        print("REQUEST BODY:", body)
    print("RESPONSE TEXT:", r.text)
    print("================================\n")

    r.raise_for_status()

def _post_with_retries(url: str, *, headers=None, data=None, params=None, label: str = "HTTP POST") -> requests.Response:
    last_err = None
    for attempt in range(POST_RETRY_MAX + 1):
        try:
            r = requests.post(url, headers=headers, data=data, params=params, timeout=HTTP_TIMEOUT)
            _raise_meta_error(r, label)
            return r
        except Exception as e:
            last_err = e
            if attempt < POST_RETRY_MAX:
                time.sleep(POST_RETRY_SLEEP)
            else:
                raise
    raise last_err  # type: ignore[misc]

# =========================
# R2 (S3)
# =========================

def r2_client():
    if not (R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and BUCKET_NAME):
        raise RuntimeError("Faltan credenciales R2/S3 (R2_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME)")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )

def save_to_r2_json(key: str, payload: dict) -> None:
    s3 = r2_client()
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body, ContentType="application/json")

def load_from_r2_json(key: str):
    s3 = r2_client()
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data)
    except Exception:
        return None

def _guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    return ".jpg"

def upload_bytes_to_r2_public(image_bytes: bytes, ext: str, prefix: str = "threads_media") -> str:
    s3 = r2_client()
    h = hashlib.sha1(image_bytes).hexdigest()[:16]
    key = f"{prefix}/{h}{ext}"

    content_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType=content_type,
    )

    url = f"{R2_PUBLIC_BASE_URL}/{key}"

    head = requests.head(url, timeout=10, allow_redirects=True)
    if head.status_code != 200:
        raise RuntimeError(f"R2 public URL no accesible (status {head.status_code}): {url}")

    ct = (head.headers.get("Content-Type") or "").lower()
    if "image" not in ct:
        raise RuntimeError(f"R2 URL no parece imagen (Content-Type={ct}): {url}")

    return url

# =========================
# RSS / ARTÍCULOS (MIX + BALANCE)
# =========================

def fetch_rss_articles() -> List[Dict[str, Any]]:
    if not feedparser:
        raise RuntimeError("Falta feedparser. Agrégalo a requirements.txt: feedparser")

    raw: List[Dict[str, Any]] = []
    for feed in RSS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:50]:
                link = getattr(e, "link", None) or ""
                title = getattr(e, "title", "") or ""
                published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
                if link:
                    raw.append({
                        "title": title.strip(),
                        "link": link.strip(),
                        "published": published,
                        "feed": feed,
                    })
        except Exception:
            continue

    # dedupe por link
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for a in raw:
        if a["link"] not in seen:
            seen.add(a["link"])
            deduped.append(a)

    # balance por feed (limita dominio dominante)
    if MAX_PER_FEED > 0:
        counts: Dict[str, int] = {}
        balanced: List[Dict[str, Any]] = []
        for a in deduped:
            f = a.get("feed") or "unknown"
            counts.setdefault(f, 0)
            if counts[f] < MAX_PER_FEED:
                balanced.append(a)
                counts[f] += 1
        deduped = balanced

    # shuffle global (mix real)
    if SHUFFLE_ARTICLES:
        random.shuffle(deduped)

    return deduped

# =========================
# EXTRAER IMÁGENES (og/twitter + fallback <img>)
# =========================

META_IMAGE_RE = re.compile(
    r'<meta\s+(?:property|name)=["\'](og:image|twitter:image)["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE
)
META_IMAGE_RE2 = re.compile(
    r'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\'](og:image|twitter:image)["\']',
    re.IGNORECASE
)
IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)

def extract_best_images(page_url: str, max_images: int = 5) -> List[str]:
    """
    Devuelve lista de URLs de imagen (1..N) con prioridad:
    1) og:image
    2) twitter:image
    3) <img src> del artículo (limitado)
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(page_url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        html = r.text

        found: List[str] = []

        # meta og/twitter
        metas: List[Tuple[str, str]] = []
        for m in META_IMAGE_RE.finditer(html):
            metas.append((m.group(1).lower(), m.group(2).strip()))
        for m in META_IMAGE_RE2.finditer(html):
            metas.append((m.group(2).lower(), m.group(1).strip()))

        for k in ["og:image", "twitter:image"]:
            for (name, img) in metas:
                if name == k and img:
                    if img.startswith("/"):
                        img = urljoin(page_url, img)
                    if img not in found:
                        found.append(img)

        # fallback: imgs del HTML
        if len(found) < max_images:
            for m in IMG_SRC_RE.finditer(html):
                img = m.group(1).strip()
                if not img:
                    continue
                if img.startswith("/"):
                    img = urljoin(page_url, img)
                if img.lower().startswith("data:"):
                    continue
                if img not in found:
                    found.append(img)
                if len(found) >= max_images:
                    break

        return found[:max_images]
    except Exception:
        return []

def extract_og_image(url: str) -> Optional[str]:
    imgs = extract_best_images(url, max_images=1)
    return imgs[0] if imgs else None

def download_image_bytes(image_url: str) -> Tuple[bytes, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": image_url,
    }
    r = requests.get(image_url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    ext = _guess_ext_from_content_type(r.headers.get("Content-Type", ""))
    return r.content, ext

# =========================
# OPENAI: GENERAR TEXTO
# =========================

def openai_client():
    if not OpenAI:
        raise RuntimeError("No se pudo importar OpenAI. Revisa requirements.txt (openai).")
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_text(prompt: str) -> str:
    client = openai_client()

    # Responses API (si está disponible)
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass

    # Fallback chat.completions
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

def build_threads_text(item: Dict[str, Any], mode: str = "new") -> str:
    title = item.get("title", "")
    link = item.get("link", "")

    if mode == "repost":
        prompt = f"""
Eres editor para una cuenta de Threads sobre esports/gaming.
Reescribe este post como un REPOST con otra mirada/opinión, sin sonar repetido.
- 1 párrafo corto, máximo 260 caracteres si es posible.
- Termina con una pregunta para la comunidad.
- Incluye "Fuente:" + link.
Datos:
Título: {title}
Link: {link}
"""
    else:
        prompt = f"""
Eres editor para una cuenta de Threads sobre esports/gaming.
Crea un post:
- 1 párrafo corto (máximo 260-320 caracteres).
- Termina con una pregunta a la comunidad.
- Incluye "Fuente:" + link.
Datos:
Título: {title}
Link: {link}
"""
    text = openai_text(prompt)
    if "Fuente:" not in text:
        text = f"{text}\n\nFuente: {link}"
    return text.strip()

def build_instagram_caption(item: Dict[str, Any], mode: str, link: str) -> str:
    title = item.get("title", "")
    if mode == "repost":
        prompt = f"""
Eres editor de Instagram (esports/gaming).
Escribe un caption natural y humano:
- 1-2 párrafos cortos
- 5-10 hashtags relevantes al final
- Cierra con una pregunta
- Incluye "Fuente:" + link al final
Título: {title}
Link: {link}
"""
    else:
        prompt = f"""
Eres editor de Instagram (esports/gaming).
Escribe un caption natural y humano:
- 1-2 párrafos cortos
- 5-10 hashtags relevantes al final
- Cierra con una pregunta
- Incluye "Fuente:" + link al final
Título: {title}
Link: {link}
"""
    text = openai_text(prompt).strip()
    if "Fuente:" not in text:
        text = f"{text}\n\nFuente: {link}"
    return text

# =========================
# THREADS API + WAIT (evita Media Not Found)
# =========================

def threads_create_container_image(user_id: str, access_token: str, text: str, image_url: str) -> str:
    url = f"{THREADS_GRAPH}/{user_id}/threads"
    data = {"media_type": "IMAGE", "image_url": image_url, "text": text}
    r = _post_with_retries(url, headers=_threads_headers(access_token), data=data, label="THREADS CREATE_CONTAINER")
    return r.json()["id"]

def threads_wait_container(container_id: str, access_token: str, timeout_sec: Optional[int] = None) -> Dict[str, Any]:
    if timeout_sec is None:
        timeout_sec = CONTAINER_WAIT_TIMEOUT

    url = f"{THREADS_GRAPH}/{container_id}"
    start = time.time()
    last = None

    while time.time() - start < timeout_sec:
        r = requests.get(
            url,
            headers=_threads_headers(access_token),
            params={"fields": "status,error_message"},
            timeout=HTTP_TIMEOUT
        )
        _raise_meta_error(r, "THREADS CONTAINER STATUS")

        j = r.json()
        last = j
        status = j.get("status")

        if status == "FINISHED":
            return j
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"Container failed: {j}")

        time.sleep(CONTAINER_POLL_INTERVAL)

    raise TimeoutError(f"Container not ready after {timeout_sec}s: {last}")

def threads_publish(user_id: str, access_token: str, container_id: str) -> Dict[str, Any]:
    url = f"{THREADS_GRAPH}/{user_id}/threads_publish"
    r = _post_with_retries(
        url,
        headers=_threads_headers(access_token),
        data={"creation_id": container_id},
        label="THREADS PUBLISH"
    )
    return r.json()

def threads_publish_text_image(text: str, image_url_from_news: str) -> Dict[str, Any]:
    if THREADS_DRY_RUN:
        print("[DRY_RUN] Threads post:", text)
        print("[DRY_RUN] Image source:", image_url_from_news)
        return {"ok": True, "dry_run": True}

    if not THREADS_USER_ACCESS_TOKEN:
        raise RuntimeError("Falta THREADS_USER_ACCESS_TOKEN")

    img_bytes, ext = download_image_bytes(image_url_from_news)
    r2_url = upload_bytes_to_r2_public(img_bytes, ext)
    print("IMAGE re-hosted on R2:", r2_url)

    container_id = threads_create_container_image(THREADS_USER_ID, THREADS_USER_ACCESS_TOKEN, text, r2_url)
    print("Container created:", container_id)

    threads_wait_container(container_id, THREADS_USER_ACCESS_TOKEN)

    res = threads_publish(THREADS_USER_ID, THREADS_USER_ACCESS_TOKEN, container_id)
    print("Threads publish response:", res)

    return {"ok": True, "container": {"id": container_id}, "publish": res, "image_url": r2_url}

# =========================
# STATE (anti-duplicado + repost)
# =========================

def load_threads_state() -> Dict[str, Any]:
    state = load_from_r2_json(THREADS_STATE_KEY)
    if not state:
        state = {
            "posted_items": {},  # link -> {last_posted_at, times}
            "posted_links": [],  # legacy
            "last_posted_at": None,
        }
    state.setdefault("posted_items", {})
    state.setdefault("posted_links", [])
    state.setdefault("last_posted_at", None)
    return state

def save_threads_state(state: Dict[str, Any]) -> None:
    save_to_r2_json(THREADS_STATE_KEY, state)

def mark_posted(state: Dict[str, Any], link: str) -> None:
    pi = state["posted_items"].get(link, {"times": 0, "last_posted_at": None})
    pi["times"] = int(pi.get("times", 0)) + 1
    pi["last_posted_at"] = iso_now()
    state["posted_items"][link] = pi

    if link not in state["posted_links"]:
        state["posted_links"].append(link)
    state["posted_links"] = state["posted_links"][-400:]
    state["last_posted_at"] = iso_now()

# =========================
# SELECCIÓN: NUEVO vs REPOST
# =========================

def is_new_allowed(state: Dict[str, Any], link: str) -> bool:
    return link not in state["posted_items"]

def repost_eligible(state: Dict[str, Any], link: str) -> bool:
    pi = state["posted_items"].get(link)
    if not pi:
        return False
    times = int(pi.get("times", 0))
    if times >= REPOST_MAX_TIMES:
        return False
    last = pi.get("last_posted_at")
    if not last:
        return True
    return days_since(last) >= REPOST_WINDOW_DAYS

def pick_item(articles: List[Dict[str, Any]], state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    for a in articles:
        if a.get("link") and is_new_allowed(state, a["link"]):
            return a, "new"

    if REPOST_ENABLE:
        for a in articles:
            if a.get("link") and a["link"] in state["posted_items"] and repost_eligible(state, a["link"]):
                return a, "repost"

    return None, "none"

# =========================
# IG QUEUE HELPERS
# =========================

def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def choose_ig_format(has_video: bool, image_count: int) -> str:
    # prioridad reels (solo si hay video), luego carrusel, luego imagen
    if has_video:
        return "reel"
    if image_count >= 2:
        return "carousel"
    return "image"

def save_ig_queue_item(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    h = _short_hash(s + iso_now())
    key = f"{IG_QUEUE_PREFIX}/{now_utc().strftime('%Y%m%d_%H%M%S')}_{h}.json"
    save_to_r2_json(key, payload)
    return key

# =========================
# MAIN RUN
# =========================

def run_robot_once() -> Dict[str, Any]:
    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles()

    feeds_in_run = sorted(set([a.get("feed", "") for a in articles]))
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={MAX_PER_FEED}, SHUFFLE={SHUFFLE_ARTICLES})")
    print("FEEDS EN ESTA CORRIDA:", feeds_in_run)
    print("TOTAL FEEDS:", len(feeds_in_run))

    processed: List[Dict[str, Any]] = []
    for a in articles[:MAX_AI_ITEMS]:
        processed.append(a)
        time.sleep(SLEEP_BETWEEN_ITEMS_SEC)

    state = load_threads_state()
    print("STATE posted_items:", len(state.get("posted_items", {})))

    posted_count = 0
    results: List[Dict[str, Any]] = []

    while posted_count < THREADS_AUTO_POST_LIMIT:
        item, mode = pick_item(processed, state)
        if not item:
            print("No hay item nuevo ni repost elegible.")
            break

        link = item["link"]
        label = "NUEVO" if mode == "new" else "REPOST"
        print(f"Seleccionado ({label}): {link}")

        text = build_threads_text(item, mode=mode)

        og_image = extract_og_image(link)
        if not og_image:
            print("No se encontró imagen (og/twitter/img). Se omite para evitar post sin imagen.")
            processed = [x for x in processed if x.get("link") != link]
            continue

        if not THREADS_AUTO_POST:
            print("THREADS_AUTO_POST desactivado. (No se publica)")
            results.append({"link": link, "mode": mode, "posted": False, "reason": "auto_post_off"})
            break

        print(f"Publicando en Threads ({label})...")
        try:
            res = threads_publish_text_image(text, og_image)
            mark_posted(state, link)
            save_threads_state(state)

            posted_count += 1
            results.append({"link": link, "mode": mode, "posted": True, "threads": res})
            print("Auto-post Threads: OK ✅")

            # --- IG QUEUE (reels > carousel > image) ---
            try:
                candidate_images = extract_best_images(link, max_images=IG_CAROUSEL_MAX_IMAGES)

                r2_images: List[str] = []
                for img_url in candidate_images:
                    try:
                        b, ext = download_image_bytes(img_url)
                        r2_img = upload_bytes_to_r2_public(b, ext)
                        if r2_img not in r2_images:
                            r2_images.append(r2_img)
                    except Exception:
                        continue
                    if len(r2_images) >= IG_CAROUSEL_MAX_IMAGES:
                        break

                # Si aún no hay ninguna imagen, al menos usa la de Threads rehost (si existe)
                threads_img = None
                if isinstance(res, dict):
                    threads_img = res.get("image_url")
                if not r2_images and threads_img:
                    r2_images = [threads_img]

                # Reels reales requieren video; por ahora no inventamos.
                has_video = False
                reel_video_url = None

                ig_format = choose_ig_format(has_video=has_video, image_count=len(r2_images))
                ig_caption = build_instagram_caption(item, mode=mode, link=link)

                ig_payload = {
                    "created_at": iso_now(),
                    "source": {
                        "link": link,
                        "feed": item.get("feed"),
                        "mode": mode,
                        "title": item.get("title"),
                    },
                    "format": ig_format,  # "reel" | "carousel" | "image"
                    "caption": ig_caption,
                    "assets": {
                        "images": r2_images[:IG_CAROUSEL_MAX_IMAGES],
                        "reel_video_url": reel_video_url,
                    },
                    "threads": {
                        "publish_id": (res.get("publish") or {}).get("id") if isinstance(res, dict) else None,
                        "image_url": threads_img,
                    },
                }

                ig_key = save_ig_queue_item(ig_payload)
                print("IG queue guardado en R2:", ig_key)
            except Exception as e:
                print("IG queue: falló (no rompe el run):", str(e))

        except Exception as e:
            print("Auto-post Threads: FALLÓ ❌")
            print("ERROR:", str(e))
            results.append({"link": link, "mode": mode, "posted": False, "error": str(e)})
            break

        processed = [x for x in processed if x.get("link") != link]

    return {"posted_count": posted_count, "results": results}

def save_run_payload(payload: Dict[str, Any]) -> None:
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    key = f"editorial_run_{run_id}.json"
    save_to_r2_json(key, payload)
    print("Archivo guardado en R2:", key)

if __name__ == "__main__":
    result = run_robot_once()
    payload = {
        "generated_at": iso_now(),
        "mix": {
            "shuffle": SHUFFLE_ARTICLES,
            "max_per_feed": MAX_PER_FEED,
            "max_ai_items": MAX_AI_ITEMS,
        },
        "threads_auto_post": {
            "enabled": THREADS_AUTO_POST,
            "limit": THREADS_AUTO_POST_LIMIT,
            "dry_run": THREADS_DRY_RUN,
            "repost_enable": REPOST_ENABLE,
            "repost_max_times": REPOST_MAX_TIMES,
            "repost_window_days": REPOST_WINDOW_DAYS,
        },
        "ig_queue": {
            "prefix": IG_QUEUE_PREFIX,
            "carousel_max_images": IG_CAROUSEL_MAX_IMAGES,
        },
        "result": result,
    }

    save_run_payload(payload)
    print("RUN COMPLETED")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
