import os
import re
import json
import time
import hashlib
import random
import tempfile
import subprocess
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, parse_qs
from typing import Optional, Tuple, Dict, Any, List

import requests
import boto3

# RSS
try:
    import feedparser
except Exception:
    feedparser = None

print("RUNNING MEDIA ENGINE (Threads REAL + IG REEL AUTO + IG PUBLISH + Multi-account via accounts.json)")


# =========================
# Helpers: env safe
# =========================
def env_nonempty(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return int(v.strip())
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


# =========================
# MODE SWITCH (A or B)
# =========================
RUN_MODE = (env_nonempty("RUN_MODE", "A") or "A").strip().upper()
if RUN_MODE in ("B", "UGC", "MODE_B", "MODEB"):
    print(">>> RUN_MODE=B detectado. Corriendo Modo B (UGC) y saliendo.")
    from ugc_mode_b import run_mode_b
    run_mode_b()
    raise SystemExit(0)


# =========================
# GLOBAL ENV
# =========================
OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL") or "").rstrip("/")

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

# Threads
ENABLE_THREADS_PUBLISH = env_bool("ENABLE_THREADS_PUBLISH", True)
THREADS_GRAPH = (env_nonempty("THREADS_GRAPH", "https://graph.threads.net") or "https://graph.threads.net").rstrip("/")
THREADS_USER_ACCESS_TOKEN = env_nonempty("THREADS_USER_ACCESS_TOKEN")

# Instagram
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", True)
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
IG_USER_ID = env_nonempty("IG_USER_ID")
GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

# Reels
ENABLE_REELS = env_bool("ENABLE_REELS", True)
REEL_SECONDS = env_int("REEL_SECONDS", 15)
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)

ASSET_BG = env_nonempty("ASSET_BG", "assets/bg.jpg")
ASSET_LOGO = env_nonempty("ASSET_LOGO", "assets/logo.png")
ASSET_MUSIC = env_nonempty("ASSET_MUSIC", "assets/music.mp3")

DRY_RUN = env_bool("DRY_RUN", False)

# Gamer Reel Engine
from reel_gamer_engine import render_gamer_reel_mp4_bytes  # should_add_voice lo activamos luego


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
# HTTP helpers
# =========================
def _raise_meta_error(r: requests.Response, label: str = "HTTP") -> None:
    if r.status_code < 400:
        return
    print(f"\n====== {label} ERROR ======")
    print("URL:", r.request.url)
    print("METHOD:", r.request.method)
    if r.request.body:
        body = r.request.body
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        print("REQUEST BODY:", body)
    print("STATUS:", r.status_code)
    print("RESPONSE TEXT:", r.text)
    print("========================\n")
    r.raise_for_status()

def _threads_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def _post(url: str, *, headers=None, data=None, params=None, label: str = "HTTP POST") -> requests.Response:
    r = requests.post(url, headers=headers, data=data, params=params, timeout=HTTP_TIMEOUT)
    _raise_meta_error(r, label)
    return r


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

def s3_put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3 = r2_client()
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)

def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3_put_bytes(key, b, "application/json")

def s3_get_bytes(key: str) -> bytes:
    s3 = r2_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return obj["Body"].read()

def s3_get_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        b = s3_get_bytes(key)
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None

def r2_public_url(key: str) -> str:
    base = (env_nonempty("R2_PUBLIC_BASE_URL", R2_PUBLIC_BASE_URL) or "").rstrip("/")
    if not base.startswith("http"):
        raise RuntimeError("R2_PUBLIC_BASE_URL inválido o vacío (debe empezar por https://)")
    return f"{base}/{key}"

def upload_image_bytes_to_r2(image_bytes: bytes, ext: str, prefix: str) -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    content_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext.lower(), "image/jpeg")
    h = hashlib.sha1(image_bytes).hexdigest()[:16]
    prefix = (prefix or "").strip().strip("/")
    key = f"{prefix}/{h}{ext}" if prefix else f"{h}{ext}"
    s3_put_bytes(key, image_bytes, content_type)
    return r2_public_url(key)

def upload_video_bytes_to_r2(video_bytes: bytes, prefix: str) -> str:
    h = hashlib.sha1(video_bytes).hexdigest()[:16]
    prefix = (prefix or "").strip().strip("/")
    key = f"{prefix}/{h}.mp4" if prefix else f"{h}.mp4"
    s3_put_bytes(key, video_bytes, "video/mp4")
    return r2_public_url(key)


# =========================
# RSS / HTML image extraction (+ YouTube thumbnails patch)
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

def normalize_url(maybe_url: str, base_url: str) -> Optional[str]:
    if not maybe_url:
        return None
    u = (maybe_url or "").strip()
    if not u:
        return None
    if u.startswith("//"):
        base = urlparse(base_url)
        scheme = base.scheme or "https"
        u = f"{scheme}:{u}"
    parsed = urlparse(u)
    if not parsed.scheme or not parsed.netloc:
        u = urljoin(base_url, u)
    parsed2 = urlparse(u)
    if not parsed2.scheme or not parsed2.netloc:
        return None
    return u

def youtube_video_id_from_url(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if "youtu.be" in host:
            vid = (p.path or "").strip("/").split("/")[0]
            return vid or None
        if "youtube.com" in host or "m.youtube.com" in host:
            qs = parse_qs(p.query or "")
            if "v" in qs and qs["v"]:
                return qs["v"][0]
            # /shorts/<id>
            parts = (p.path or "").split("/")
            if "shorts" in parts:
                i = parts.index("shorts")
                if i + 1 < len(parts):
                    return parts[i + 1]
        return None
    except Exception:
        return None

def youtube_thumbnail_urls(video_id: str) -> List[str]:
    # maxres may not exist; fallback
    return [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
    ]

def extract_best_images(page_url: str, max_images: int = 5) -> List[str]:
    # PATCH: If YouTube link, return thumbnails immediately
    vid = youtube_video_id_from_url(page_url)
    if vid:
        return youtube_thumbnail_urls(vid)[:max_images]

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(page_url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        html = r.text

        found: List[str] = []
        metas: List[Tuple[str, str]] = []

        for m in META_IMAGE_RE.finditer(html):
            metas.append((m.group(1).lower(), m.group(2).strip()))
        for m in META_IMAGE_RE2.finditer(html):
            metas.append((m.group(2).lower(), m.group(1).strip()))

        for k in ["og:image", "twitter:image"]:
            for (name, img) in metas:
                if name == k and img:
                    img2 = normalize_url(img, page_url)
                    if img2 and img2 not in found:
                        found.append(img2)

        if len(found) < max_images:
            for m in IMG_SRC_RE.finditer(html):
                img = m.group(1).strip()
                if not img or img.lower().startswith("data:"):
                    continue
                img2 = normalize_url(img, page_url)
                if img2 and img2 not in found:
                    found.append(img2)
                if len(found) >= max_images:
                    break

        return found[:max_images]
    except Exception:
        return []

def _guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    return ".jpg"

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

def fetch_rss_articles(rss_feeds: List[str], max_per_feed: int, shuffle: bool) -> List[Dict[str, Any]]:
    if not feedparser:
        raise RuntimeError("Falta feedparser. Agrégalo a requirements.txt: feedparser")

    raw: List[Dict[str, Any]] = []
    for feed in rss_feeds:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:50]:
                link = getattr(e, "link", None) or ""
                title = getattr(e, "title", "") or ""
                published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
                if link:
                    raw.append({"title": title.strip(), "link": link.strip(), "published": published, "feed": feed})
        except Exception:
            continue

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for a in raw:
        if a["link"] not in seen:
            seen.add(a["link"])
            deduped.append(a)

    if max_per_feed > 0:
        counts: Dict[str, int] = {}
        balanced: List[Dict[str, Any]] = []
        for a in deduped:
            f = a.get("feed") or "unknown"
            counts.setdefault(f, 0)
            if counts[f] < max_per_feed:
                balanced.append(a)
                counts[f] += 1
        deduped = balanced

    if shuffle:
        random.shuffle(deduped)

    return deduped


# =========================
# OpenAI (text) via HTTP
# =========================
def openai_text(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")
    model = OPENAI_MODEL or "gpt-4.1-mini"
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        # fallback
        url2 = "https://api.openai.com/v1/chat/completions"
        payload2 = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        r2 = requests.post(url2, headers=headers, json=payload2, timeout=60)
        r2.raise_for_status()
        j2 = r2.json()
        return (j2["choices"][0]["message"]["content"] or "").strip()

    j = r.json()
    out = j.get("output_text")
    if out:
        return str(out).strip()

    texts = []
    for c in j.get("output", []) or []:
        for part in c.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                texts.append(part["text"])
    return "\n".join(texts).strip()


# =========================
# Copy (gaming esports)
# =========================
THREADS_PROMPT_ESPORTS = """Eres editor para una cuenta de Threads sobre esports/gaming en español LATAM.
Crea un post:
- 1 párrafo corto, claro y con vibe esports.
- Termina con una pregunta a la comunidad.
- Incluye "Fuente:" + link al final.
Datos:
Título: {title}
Link: {link}
"""

IG_PROMPT_ESPORTS = """Eres editor de Instagram (esports/gaming) en español LATAM.
Escribe un caption natural y humano:
- 1-2 párrafos cortos
- 5-10 hashtags relevantes al final
- Cierra con una pregunta
- Incluye "Fuente:" + link al final
Título: {title}
Link: {link}
"""

def build_threads_text(item: Dict[str, Any], mode: str = "new") -> str:
    title = item.get("title", "")
    link = item.get("link", "")
    prompt = THREADS_PROMPT_ESPORTS.format(title=title, link=link)
    if mode == "repost":
        prompt += "\nExtra: reescribe como REPOST con otra mirada/opinión, sin sonar repetido."
    text = openai_text(prompt).strip()
    if "Fuente:" not in text:
        text = f"{text}\n\nFuente: {link}"
    return text.strip()

def build_instagram_caption(item: Dict[str, Any], link: str) -> str:
    title = item.get("title", "")
    prompt = IG_PROMPT_ESPORTS.format(title=title, link=link)
    text = openai_text(prompt).strip()
    if "Fuente:" not in text:
        text = f"{text}\n\nFuente: {link}"
    return text.strip()

def clip_threads_text(text: str, max_chars: int = 500) -> str:
    text = (text or "").strip()
    link = ""
    if "Fuente:" in text:
        parts = text.split("Fuente:", 1)
        body = parts[0].strip()
        link = "Fuente:" + parts[1].strip()
        reserve = min(len(link) + 1, max_chars)
        body_max = max_chars - reserve
        body = body[:max(0, body_max)].rstrip()
        text = (body + "\n" + link).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


# =========================
# Threads API
# =========================
def threads_create_container_image(user_id: str, access_token: str, text: str, image_url: str) -> str:
    url = f"{THREADS_GRAPH}/{user_id}/threads"
    text = clip_threads_text(text, 500)
    data = {"media_type": "IMAGE", "image_url": image_url, "text": text}
    r = _post(url, headers=_threads_headers(access_token), data=data, label="THREADS CREATE_CONTAINER")
    return r.json()["id"]

def threads_wait_container(container_id: str, access_token: str, timeout_sec: int = 120) -> None:
    url = f"{THREADS_GRAPH}/{container_id}"
    start = time.time()
    last = None
    while time.time() - start < timeout_sec:
        r = requests.get(url, headers=_threads_headers(access_token), params={"fields": "status,error_message"}, timeout=HTTP_TIMEOUT)
        _raise_meta_error(r, "THREADS CONTAINER STATUS")
        j = r.json()
        last = j
        status = j.get("status")
        if status == "FINISHED":
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"Container failed: {j}")
        time.sleep(2)
    raise TimeoutError(f"Container not ready after {timeout_sec}s: {last}")

def threads_publish(user_id: str, access_token: str, container_id: str) -> Dict[str, Any]:
    url = f"{THREADS_GRAPH}/{user_id}/threads_publish"
    r = _post(url, headers=_threads_headers(access_token), data={"creation_id": container_id}, label="THREADS PUBLISH")
    return r.json()


# =========================
# IG Graph API
# =========================
def ig_api_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.post(url, data=data, timeout=HTTP_TIMEOUT)
    _raise_meta_error(r, f"IG POST {path}")
    return r.json()

def ig_wait_container(creation_id: str, timeout_sec: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        j = requests.get(
            f"{GRAPH_BASE}/{creation_id}",
            params={"fields": "status_code", "access_token": IG_ACCESS_TOKEN},
            timeout=HTTP_TIMEOUT,
        )
        _raise_meta_error(j, "IG CONTAINER STATUS")
        data = j.json()
        status = (data.get("status_code") or "").upper()
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {data}")
        time.sleep(3)
    raise TimeoutError(f"IG container not ready after {timeout_sec}s")

def ig_publish_reel(video_url: str, caption: str) -> Dict[str, Any]:
    if not (IG_USER_ID and IG_ACCESS_TOKEN):
        raise RuntimeError("Faltan IG_USER_ID / IG_ACCESS_TOKEN")

    j = ig_api_post(
        f"{IG_USER_ID}/media",
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": IG_ACCESS_TOKEN,
        },
    )
    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG reels create failed: {j}")
    ig_wait_container(creation_id, timeout_sec=900)
    res = ig_api_post(f"{IG_USER_ID}/media_publish", {"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN})
    return res


# =========================
# State
# =========================
def load_state(state_key: str) -> Dict[str, Any]:
    st = s3_get_json(state_key) or {}
    st.setdefault("posted_items", {})
    st.setdefault("posted_links", [])
    st.setdefault("last_posted_at", None)
    return st

def save_state(state_key: str, st: Dict[str, Any]) -> None:
    s3_put_json(state_key, st)

def mark_posted(st: Dict[str, Any], link: str) -> None:
    pi = st["posted_items"].get(link, {"times": 0, "last_posted_at": None})
    pi["times"] = int(pi.get("times", 0)) + 1
    pi["last_posted_at"] = iso_now()
    st["posted_items"][link] = pi
    if link not in st["posted_links"]:
        st["posted_links"].append(link)
    st["posted_links"] = st["posted_links"][-400:]
    st["last_posted_at"] = iso_now()

def is_new_allowed(st: Dict[str, Any], link: str) -> bool:
    return link not in st["posted_items"]

def repost_eligible(st: Dict[str, Any], link: str, repost_max_times: int, repost_window_days: int) -> bool:
    pi = st["posted_items"].get(link)
    if not pi:
        return False
    times = int(pi.get("times", 0))
    if times >= repost_max_times:
        return False
    last = pi.get("last_posted_at")
    if not last:
        return True
    return days_since(last) >= repost_window_days

def pick_item(articles: List[Dict[str, Any]], st: Dict[str, Any], repost_enable: bool, repost_max_times: int, repost_window_days: int) -> Tuple[Optional[Dict[str, Any]], str]:
    for a in articles:
        if a.get("link") and is_new_allowed(st, a["link"]):
            return a, "new"
    if repost_enable:
        for a in articles:
            if a.get("link") and a["link"] in st["posted_items"] and repost_eligible(st, a["link"], repost_max_times, repost_window_days):
                return a, "repost"
    return None, "none"


# =========================
# Accounts
# =========================
def load_accounts() -> List[Dict[str, Any]]:
    with open("accounts.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "accounts" in data and isinstance(data["accounts"], list):
        return data["accounts"]
    if isinstance(data, list):
        return data
    return []


# =========================
# Main per account run (MODE A)
# =========================
def run_account(cfg: Dict[str, Any]) -> Dict[str, Any]:
    account_id = cfg.get("account_id", "unknown")
    print(f"\n===== RUN ACCOUNT: {account_id} =====")

    rss_feeds = cfg.get("rss_feeds") or []
    max_per_feed = int(cfg.get("max_per_feed", 3))
    shuffle = bool(cfg.get("shuffle_articles", True))
    max_ai_items = int(cfg.get("max_ai_items", 20))

    assets_cfg = cfg.get("assets", {}) or {}
    asset_bg = assets_cfg.get("bg") or ASSET_BG
    asset_logo = assets_cfg.get("logo") or ASSET_LOGO
    asset_music = assets_cfg.get("music") or ASSET_MUSIC
    cta_text = assets_cfg.get("cta") or "Sigue para más"

    threads_cfg = cfg.get("threads", {}) or {}
    threads_user_id = threads_cfg.get("user_id", "me")
    state_key = threads_cfg.get("state_key", f"accounts/{account_id}/threads_state.json")
    auto_post_limit = int(threads_cfg.get("auto_post_limit", 1))
    repost_enable = bool(threads_cfg.get("repost_enable", True))
    repost_max_times = int(threads_cfg.get("repost_max_times", 3))
    repost_window_days = int(threads_cfg.get("repost_window_days", 7))

    r2_cfg = cfg.get("r2", {}) or {}
    threads_media_prefix = (r2_cfg.get("threads_media_prefix") or f"threads_media/{account_id}").strip().strip("/")
    reels_prefix = (r2_cfg.get("reels_prefix") or f"ugc/reels/{account_id}").strip().strip("/")

    if auto_post_limit <= 0 or not rss_feeds:
        print(f"Cuenta {account_id} está apagada (auto_post_limit=0 o rss_feeds vacío). Saltando ✅")
        return {"generated_at": iso_now(), "account_id": account_id, "skipped": True, "reason": "disabled_or_no_feeds"}

    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles(rss_feeds, max_per_feed=max_per_feed, shuffle=shuffle)
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={max_per_feed}, SHUFFLE={shuffle})")

    state = load_state(state_key)
    print("STATE posted_items:", len(state.get("posted_items", {})))

    posted_count = 0
    results = []

    while posted_count < auto_post_limit:
        item, mode = pick_item(articles[:max_ai_items], state, repost_enable, repost_max_times, repost_window_days)
        if not item:
            print("No hay item nuevo ni repost elegible.")
            break

        link = item["link"]
        label = "NUEVO" if mode == "new" else "REPOST"
        print(f"Seleccionado ({label}): {link}")

        # 1) Texto Threads + Caption IG
        text_threads = build_threads_text(item, mode=mode)
        caption_ig = build_instagram_caption(item, link=link)

        # 2) Imagen (con parche YouTube)
        img_candidates = extract_best_images(link, max_images=5)
        if not img_candidates:
            print("No se encontró imagen. Se omite.")
            # remove and continue
            articles = [x for x in articles if x.get("link") != link]
            continue

        chosen_img = None
        img_bytes = None
        img_ext = ".jpg"
        for u in img_candidates:
            try:
                img_bytes, img_ext = download_image_bytes(u)
                chosen_img = u
                break
            except Exception:
                continue

        if not chosen_img or not img_bytes:
            print("Todas las imágenes del artículo fallaron. Se omite.")
            articles = [x for x in articles if x.get("link") != link]
            continue

        # 3) Rehost imagen en R2
        img_r2_url = upload_image_bytes_to_r2(img_bytes, img_ext, threads_media_prefix)
        print("Imagen rehost R2:", img_r2_url)

        # 4) Publicar Threads
        threads_res = None
        if ENABLE_THREADS_PUBLISH and not DRY_RUN:
            if not THREADS_USER_ACCESS_TOKEN:
                raise RuntimeError("Falta THREADS_USER_ACCESS_TOKEN")
            container_id = threads_create_container_image(threads_user_id, THREADS_USER_ACCESS_TOKEN, text_threads, img_r2_url)
            print("Container created:", container_id)
            threads_wait_container(container_id, THREADS_USER_ACCESS_TOKEN, timeout_sec=180)
            pub = threads_publish(threads_user_id, THREADS_USER_ACCESS_TOKEN, container_id)
            print("Threads publish response:", pub)
            threads_res = {"ok": True, "container": {"id": container_id}, "publish": pub, "image_url": img_r2_url}
        else:
            print("[DRY_RUN] Threads text:", clip_threads_text(text_threads, 500))
            threads_res = {"ok": True, "dry_run": True, "image_url": img_r2_url}

        # 5) Generar Reel gamer dinámico (local) -> subir R2 -> publicar IG
        ig_res = None
        ig_kind = None

        if ENABLE_REELS:
            # guardamos la imagen como archivo local temporal para el motor
            with tempfile.TemporaryDirectory() as td:
                news_path = os.path.join(td, f"news{img_ext}")
                with open(news_path, "wb") as f:
                    f.write(img_bytes)

                reel_bytes, plan_used = render_gamer_reel_mp4_bytes(
                    headline=item.get("title", ""),
                    link=link,
                    news_image_path=news_path,
                    bg_path=asset_bg,
                    logo_path=asset_logo,
                    seconds=REEL_SECONDS,
                    music_path=asset_music,
                    voice_mp3_path=None,  # luego activamos voz IA
                )

            reel_url = upload_video_bytes_to_r2(reel_bytes, reels_prefix)
            print("Reel URL R2:", reel_url)
            print("Plan usado:", (plan_used.get("style") or {}).get("name"), "| logo_mode:", plan_used.get("logo_mode"))

            if ENABLE_IG_PUBLISH and not DRY_RUN:
                ig_kind = "reel"
                ig_res = ig_publish_reel(video_url=reel_url, caption=caption_ig)
                print("IG publish OK:", ig_res)
            else:
                print("[DRY_RUN] IG reel:", reel_url)
                print("[DRY_RUN] caption:", caption_ig[:500])
                ig_res = {"ok": True, "dry_run": True, "video_url": reel_url}
                ig_kind = "reel"

        # 6) Mark posted
        mark_posted(state, link)
        save_state(state_key, state)
        posted_count += 1

        results.append(
            {
                "link": link,
                "mode": mode,
                "threads": threads_res,
                "ig": ig_res,
                "ig_kind": ig_kind,
                "dry_run": DRY_RUN,
            }
        )

        break

    payload = {
        "generated_at": iso_now(),
        "account_id": account_id,
        "mix": {"shuffle": shuffle, "max_per_feed": max_per_feed, "max_ai_items": max_ai_items},
        "settings": {
            "enable_reels": ENABLE_REELS,
            "reel_seconds": REEL_SECONDS,
            "enable_ig_publish": ENABLE_IG_PUBLISH,
            "enable_threads_publish": ENABLE_THREADS_PUBLISH,
            "dry_run": DRY_RUN,
            "graph_version": f"v{GRAPH_VERSION}",
        },
        "result": {"posted_count": posted_count, "results": results},
    }
    return payload


def save_run_payload(account_id: str, payload: Dict[str, Any]) -> str:
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    key = f"accounts/{account_id}/runs/editorial_run_{run_id}.json"
    s3_put_json(key, payload)
    print("Archivo guardado en R2:", key)
    return key


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    accounts = load_accounts()
    if not accounts:
        raise RuntimeError("No se encontraron cuentas. Falta accounts.json o está vacío.")

    print("ENV CHECK:")
    print(" - RUN_MODE:", RUN_MODE)
    print(" - DRY_RUN:", DRY_RUN)
    print(" - ENABLE_THREADS_PUBLISH:", ENABLE_THREADS_PUBLISH)
    print(" - ENABLE_IG_PUBLISH:", ENABLE_IG_PUBLISH)
    print(" - ENABLE_REELS:", ENABLE_REELS)
    print(" - R2_PUBLIC_BASE_URL:", (R2_PUBLIC_BASE_URL or "")[:80])
    print(" - OPENAI_MODEL:", OPENAI_MODEL)
    print(" - GRAPH_BASE:", GRAPH_BASE)
    print(" - IG_USER_ID set:", bool(IG_USER_ID))
    print(" - IG_ACCESS_TOKEN set:", bool(IG_ACCESS_TOKEN))

    all_results = []
    for cfg in accounts:
        account_id = cfg.get("account_id", "unknown")
        payload = run_account(cfg)
        run_key = save_run_payload(account_id, payload)
        all_results.append({"account_id": account_id, "run_key": run_key, "payload": payload})
        print("RUN COMPLETED:", account_id)

    print("\n===== SUMMARY =====")
    print(json.dumps({"runs": all_results}, ensure_ascii=False, indent=2))
