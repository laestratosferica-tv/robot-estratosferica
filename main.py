# main.py
# Robot Editorial + Auto-post Threads con re-host de imágenes en R2
# PROD v6: auto-mix feeds (shuffle + max per feed) + Threads WAIT + state + repost

import os
import re
import json
import time
import hashlib
import random
from datetime import datetime, timezone
from urllib.parse import urljoin

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


print("RUNNING MULTIRED v6 (PROD: AUTO-MIX FEEDS)")

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

THREADS_USER_ID = os.getenv("THREADS_USER_ID", "me")
THREADS_USER_ACCESS_TOKEN = os.getenv("THREADS_USER_ACCESS_TOKEN")

THREADS_STATE_KEY = os.getenv("THREADS_STATE_KEY", "threads_state.json")

THREADS_AUTO_POST = os.getenv("THREADS_AUTO_POST", "true").lower() == "true"
THREADS_AUTO_POST_LIMIT = int(os.getenv("THREADS_AUTO_POST_LIMIT", "1"))
THREADS_DRY_RUN = os.getenv("THREADS_DRY_RUN", "false").lower() == "true"

REPOST_MAX_TIMES = int(os.getenv("REPOST_MAX_TIMES", "3"))
REPOST_WINDOW_DAYS = int(os.getenv("REPOST_WINDOW_DAYS", "7"))
REPOST_ENABLE = os.getenv("REPOST_ENABLE", "true").lower() == "true"

RSS_FEEDS = [x.strip() for x in (os.getenv("RSS_FEEDS") or "").split(",") if x.strip()]
if not RSS_FEEDS:
    RSS_FEEDS = ["https://www.dexerto.com/feed/"]

MAX_AI_ITEMS = int(os.getenv("MAX_AI_ITEMS", "5"))
SLEEP_BETWEEN_ITEMS_SEC = float(os.getenv("SLEEP_BETWEEN_ITEMS_SEC", "0.25"))

# ✅ Threads host correcto (SIN /v20.0)
THREADS_GRAPH = os.getenv("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")

# HTTP tuning
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
POST_RETRY_MAX = int(os.getenv("POST_RETRY_MAX", "2"))
POST_RETRY_SLEEP = float(os.getenv("POST_RETRY_SLEEP", "2"))

CONTAINER_WAIT_TIMEOUT = int(os.getenv("CONTAINER_WAIT_TIMEOUT", "120"))
CONTAINER_POLL_INTERVAL = float(os.getenv("CONTAINER_POLL_INTERVAL", "2"))

# ✅ Auto-mix feeds
# Max artículos por feed para evitar que uno domine (dexerto suele publicar muchísimo)
MAX_PER_FEED = int(os.getenv("MAX_PER_FEED", "10"))
SHUFFLE_ARTICLES = os.getenv("SHUFFLE_ARTICLES", "true").lower() == "true"

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

def _threads_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def _raise_meta_error(r: requests.Response, label: str = "META"):
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

def _post_with_retries(url, *, headers=None, data=None, params=None, label="HTTP POST"):
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
    raise last_err

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

def save_to_r2_json(key: str, payload: dict):
    s3 = r2_client()
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body, ContentType="application/json")

def load_from_r2_json(key: str):
    s3 = r2_client()
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data)
    except s3.exceptions.NoSuchKey:
        return None
    except Exception:
        return None

def _guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    return ".jpg"

def upload_bytes_to_r2_public(image_bytes: bytes, ext: str, prefix="threads_media") -> str:
    s3 = r2_client()
    h = hashlib.sha1(image_bytes).hexdigest()[:16]
    key = f"{prefix}/{h}{ext}"

    content_type = {
        ".png":"image/png",
        ".webp":"image/webp",
        ".jpg":"image/jpeg",
        ".jpeg":"image/jpeg",
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
# RSS / ARTÍCULOS (AUTO-MIX)
# =========================

def fetch_rss_articles():
    if not feedparser:
        raise RuntimeError("Falta feedparser. Agrégalo a requirements.txt: feedparser")

    raw = []
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

    # 1) dedupe por link (preserva orden inicial)
    seen = set()
    deduped = []
    for a in raw:
        if a["link"] not in seen:
            seen.add(a["link"])
            deduped.append(a)

    # 2) balance: máximo N por feed
    if MAX_PER_FEED > 0:
        counts = {}
        balanced = []
        for a in deduped:
            f = a.get("feed") or "unknown"
            counts.setdefault(f, 0)
            if counts[f] < MAX_PER_FEED:
                balanced.append(a)
                counts[f] += 1
        deduped = balanced

    # 3) mezclar para evitar sesgo por orden de feed
    if SHUFFLE_ARTICLES:
        random.shuffle(deduped)

    return deduped

# =========================
# EXTRAER IMAGEN OG
# =========================

OG_IMAGE_RE = re.compile(r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE)
OG_IMAGE_RE2 = re.compile(r'content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', re.IGNORECASE)

def extract_og_image(url: str) -> str | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        html = r.text
        m = OG_IMAGE_RE.search(html) or OG_IMAGE_RE2.search(html)
        if not m:
            return None
        img = m.group(1).strip()
        if img.startswith("/"):
            img = urljoin(url, img)
        return img
    except Exception:
        return None

def download_image_bytes(image_url: str) -> tuple[bytes, str]:
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

    try:
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

def build_threads_text(item: dict, mode: str = "new") -> str:
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

# =========================
# THREADS API + WAIT
# =========================

def threads_create_container_image(user_id: str, access_token: str, text: str, image_url: str) -> str:
    url = f"{THREADS_GRAPH}/{user_id}/threads"
    data = {"media_type": "IMAGE", "image_url": image_url, "text": text}
    r = _post_with_retries(url, headers=_threads_headers(access_token), data=data, label="THREADS CREATE_CONTAINER")
    return r.json()["id"]

def threads_wait_container(container_id: str, access_token: str, timeout_sec: int = None):
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

def threads_publish(user_id: str, access_token: str, container_id: str) -> dict:
    url = f"{THREADS_GRAPH}/{user_id}/threads_publish"
    r = _post_with_retries(
        url,
        headers=_threads_headers(access_token),
        data={"creation_id": container_id},
        label="THREADS PUBLISH"
    )
    return r.json()

def threads_publish_text_image(text: str, image_url_from_news: str) -> dict:
    if THREADS_DRY_RUN:
        print("[DRY_RUN] Threads post:", text)
        print("[DRY_RUN] Image source:", image_url_from_news)
        return {"ok": True, "dry_run": True}

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
# STATE + SELECCIÓN
# =========================

def load_threads_state() -> dict:
    state = load_from_r2_json(THREADS_STATE_KEY)
    if not state:
        state = {"posted_items": {}, "posted_links": [], "last_posted_at": None}
    state.setdefault("posted_items", {})
    state.setdefault("posted_links", [])
    state.setdefault("last_posted_at", None)
    return state

def save_threads_state(state: dict):
    save_to_r2_json(THREADS_STATE_KEY, state)

def mark_posted(state: dict, link: str):
    pi = state["posted_items"].get(link, {"times": 0, "last_posted_at": None})
    pi["times"] = int(pi.get("times", 0)) + 1
    pi["last_posted_at"] = iso_now()
    state["posted_items"][link] = pi

    if link not in state["posted_links"]:
        state["posted_links"].append(link)
    state["posted_links"] = state["posted_links"][-200:]
    state["last_posted_at"] = iso_now()

def is_new_allowed(state: dict, link: str) -> bool:
    return link not in state["posted_items"]

def repost_eligible(state: dict, link: str) -> bool:
    pi = state["posted_items"].get(link)
    if not pi:
        return False
    if int(pi.get("times", 0)) >= REPOST_MAX_TIMES:
        return False
    last = pi.get("last_posted_at")
    if not last:
        return True
    return days_since(last) >= REPOST_WINDOW_DAYS

def pick_item(articles: list[dict], state: dict) -> tuple[dict | None, str]:
    for a in articles:
        if a.get("link") and is_new_allowed(state, a["link"]):
            return a, "new"
    if REPOST_ENABLE:
        for a in articles:
            if a.get("link") and a["link"] in state["posted_items"] and repost_eligible(state, a["link"]):
                return a, "repost"
    return None, "none"

# =========================
# MAIN
# =========================

def run_robot_once():
    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles()
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={MAX_PER_FEED}, SHUFFLE={SHUFFLE_ARTICLES})")

    processed = []
    for a in articles[:MAX_AI_ITEMS]:
        processed.append(a)
        time.sleep(SLEEP_BETWEEN_ITEMS_SEC)

    state = load_threads_state()

    posted_count = 0
    results = []

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
            print("No se encontró og:image. Se omite para evitar post sin imagen.")
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
        except Exception as e:
            print("Auto-post Threads: FALLÓ ❌")
            print("ERROR:", str(e))
            results.append({"link": link, "mode": mode, "posted": False, "error": str(e)})
            break

        processed = [x for x in processed if x.get("link") != link]

    return {"posted_count": posted_count, "results": results}

def save_run_payload(payload: dict):
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    key = f"editorial_run_{run_id}.json"
    save_to_r2_json(key, payload)
    print("Archivo guardado en R2:", key)

if __name__ == "__main__":
    result = run_robot_once()
    payload = {
        "generated_at": iso_now(),
        "auto_mix": {"shuffle": SHUFFLE_ARTICLES, "max_per_feed": MAX_PER_FEED},
        "threads_auto_post": {
            "enabled": THREADS_AUTO_POST,
            "limit": THREADS_AUTO_POST_LIMIT,
            "dry_run": THREADS_DRY_RUN,
            "repost_enable": REPOST_ENABLE,
            "repost_max_times": REPOST_MAX_TIMES,
            "repost_window_days": REPOST_WINDOW_DAYS,
            "result": result,
        },
    }
    save_run_payload(payload)
