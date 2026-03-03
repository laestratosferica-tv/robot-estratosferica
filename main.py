import os
import re
import json
import time
import hashlib
import random
import subprocess
import tempfile
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

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

print("RUNNING MEDIA ENGINE (Threads REAL + IG REEL AUTO + IG PUBLISH + Multi-account via accounts.json)")

# =========================
# Helpers: env safe (no empty)
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
# GLOBAL ENV (infra)
# =========================

OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

# NOTE: some repos use R2_PUBLIC_BASE_URL, others use R2_PUBLIC_BASE_URL / R2_PUBLIC_BASE_URL
R2_PUBLIC_BASE_URL = env_nonempty(
    "R2_PUBLIC_BASE_URL",
    env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev")
).rstrip("/")

THREADS_GRAPH = env_nonempty("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")
THREADS_USER_ACCESS_TOKEN = env_nonempty("THREADS_USER_ACCESS_TOKEN")
THREADS_USER_ID = env_nonempty("THREADS_USER_ID", "me")

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)
POST_RETRY_MAX = env_int("POST_RETRY_MAX", 2)
POST_RETRY_SLEEP = env_float("POST_RETRY_SLEEP", 2.0)

CONTAINER_WAIT_TIMEOUT = env_int("CONTAINER_WAIT_TIMEOUT", 120)
CONTAINER_POLL_INTERVAL = env_float("CONTAINER_POLL_INTERVAL", 2.0)

VERIFY_NEWS = env_bool("VERIFY_NEWS", False)
ENABLE_TRENDS = env_bool("ENABLE_TRENDS", False)

# Reels generation
ENABLE_REELS = env_bool("ENABLE_REELS", True)
REEL_SECONDS = env_int("REEL_SECONDS", 15)
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)

DEFAULT_ASSET_BG = env_nonempty("ASSET_BG", "assets/bg.jpg")
DEFAULT_ASSET_LOGO = env_nonempty("ASSET_LOGO", "assets/logo.png")
DEFAULT_ASSET_MUSIC = env_nonempty("ASSET_MUSIC", "assets/music.mp3")  # optional single file
FONT_BOLD = env_nonempty("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Publish toggles
DRY_RUN = env_bool("DRY_RUN", False)
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", False)
ENABLE_THREADS_PUBLISH = env_bool("ENABLE_THREADS_PUBLISH", True)

# Instagram publish (Graph API)
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
IG_USER_ID = env_nonempty("IG_USER_ID")
GRAPH_VERSION = env_nonempty("GRAPH_VERSION", "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

# Runway (optional)
RUNWAY_ENABLED = env_bool("RUNWAY_ENABLED", False)
RUNWAY_API_KEY = env_nonempty("RUNWAY_API_KEY")
RUNWAY_VERSION = env_nonempty("RUNWAY_VERSION", "2024-11-06")
RUNWAY_BASE = env_nonempty("RUNWAY_BASE", "https://api.dev.runwayml.com").rstrip("/")
RUNWAY_I2V_MODEL = env_nonempty("RUNWAY_I2V_MODEL", "gen4.5")
RUNWAY_I2V_SECONDS = env_int("RUNWAY_I2V_SECONDS", 5)
RUNWAY_TIMEOUT = env_int("RUNWAY_TIMEOUT", 420)
RUNWAY_POLL_SEC = env_int("RUNWAY_POLL_SEC", 6)

# Autonomy / randomness
MUSIC_PROBABILITY = env_float("MUSIC_PROBABILITY", 0.75)  # 0..1
LOGO_PROBABILITY = env_float("LOGO_PROBABILITY", 0.85)    # 0..1
MUSIC_SEARCH_DIR = env_nonempty("MUSIC_SEARCH_DIR", "assets") or "assets"

# Better download retry for runway mp4
RUNWAY_MP4_RETRIES = env_int("RUNWAY_MP4_RETRIES", 3)

print("ENV CHECK:")
print(" - RUN_MODE:", RUN_MODE)
print(" - DRY_RUN:", DRY_RUN)
print(" - ENABLE_THREADS_PUBLISH:", ENABLE_THREADS_PUBLISH)
print(" - ENABLE_IG_PUBLISH:", ENABLE_IG_PUBLISH)
print(" - ENABLE_REELS:", ENABLE_REELS)
print(" - RUNWAY enabled:", RUNWAY_ENABLED and bool(RUNWAY_API_KEY))
print(" - R2_PUBLIC_BASE_URL set:", bool(R2_PUBLIC_BASE_URL))
print(" - OPENAI_MODEL:", OPENAI_MODEL)
print(" - GRAPH_BASE:", GRAPH_BASE)
print(" - IG_USER_ID set:", bool(IG_USER_ID))
print(" - IG_ACCESS_TOKEN set:", bool(IG_ACCESS_TOKEN))

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
# URL helpers
# =========================

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

    if u.startswith("https:///") or u.startswith("http:///"):
        base = urlparse(base_url)
        scheme = base.scheme or "https"
        host = base.netloc
        path = u.split(":///", 1)[1]
        u = f"{scheme}://{host}/{path.lstrip('/')}"

    parsed = urlparse(u)
    if not parsed.scheme or not parsed.netloc:
        u = urljoin(base_url, u)

    parsed2 = urlparse(u)
    if not parsed2.scheme or not parsed2.netloc:
        return None
    return u

def is_youtube_url(u: str) -> bool:
    try:
        p = urlparse(u)
        host = (p.netloc or "").lower()
        return "youtube.com" in host or "youtu.be" in host
    except Exception:
        return False

def extract_youtube_video_id(u: str) -> Optional[str]:
    try:
        p = urlparse(u)
        host = (p.netloc or "").lower()
        if "youtu.be" in host:
            vid = (p.path or "").strip("/").split("/")[0]
            return vid or None
        if "youtube.com" in host:
            if p.path.startswith("/watch"):
                q = parse_qs(p.query or "")
                vid = (q.get("v") or [None])[0]
                return vid
            if p.path.startswith("/shorts/"):
                return p.path.split("/shorts/", 1)[1].split("/", 1)[0] or None
            if p.path.startswith("/embed/"):
                return p.path.split("/embed/", 1)[1].split("/", 1)[0] or None
        return None
    except Exception:
        return None

def youtube_thumbnail_candidates(video_id: str) -> List[str]:
    return [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
    ]

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

def _post_with_retries(url: str, *, headers=None, data=None, params=None, json_body=None, label: str = "HTTP POST") -> requests.Response:
    last_err = None
    for attempt in range(POST_RETRY_MAX + 1):
        try:
            r = requests.post(url, headers=headers, data=data, params=params, json=json_body, timeout=HTTP_TIMEOUT)
            _raise_meta_error(r, label)
            return r
        except Exception as e:
            last_err = e
            if attempt < POST_RETRY_MAX:
                time.sleep(POST_RETRY_SLEEP)
            else:
                raise
    raise last_err  # type: ignore[misc]

def _get_with_retries(url: str, *, headers=None, params=None, timeout=30, label="HTTP GET", retries=2, allow_redirects=True) -> requests.Response:
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=allow_redirects)
            if r.status_code >= 400:
                # print meta once for last attempt
                if attempt >= retries:
                    print(f"\n====== {label} ERROR ======")
                    print("URL:", url)
                    print("STATUS:", r.status_code)
                    print("RESPONSE TEXT:", (r.text or "")[:2000])
                    print("========================\n")
                r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5)
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
    if "mp4" in ct:
        return ".mp4"
    if "mpeg" in ct or "audio" in ct:
        return ".mp3"
    return ".bin"

def upload_bytes_to_r2_public(file_bytes: bytes, ext: str, prefix: str, content_type: str, expect_kind: str = "any") -> str:
    base = (env_nonempty("R2_PUBLIC_BASE_URL", R2_PUBLIC_BASE_URL) or "").rstrip("/")
    if not base.startswith("http"):
        raise RuntimeError("R2_PUBLIC_BASE_URL inválido. Debe empezar por https://")

    s3 = r2_client()
    h = hashlib.sha1(file_bytes).hexdigest()[:16]
    prefix = (prefix or "").strip().strip("/")
    key = f"{prefix}/{h}{ext}" if prefix else f"{h}{ext}"

    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_bytes, ContentType=content_type)
    url = f"{base}/{key}"

    # best-effort check
    try:
        head = requests.head(url, timeout=10, allow_redirects=True)
        if head.status_code != 200:
            raise RuntimeError(f"R2 public URL no accesible (status {head.status_code}): {url}")
        ct = (head.headers.get("Content-Type") or "").lower()
        if expect_kind == "image" and "image" not in ct:
            raise RuntimeError(f"R2 URL no parece imagen (Content-Type={ct}): {url}")
        if expect_kind == "video" and "video" not in ct:
            print(f"AVISO: Content-Type inesperado para video: {ct} (URL {url})")
    except Exception as e:
        print("AVISO: validación HEAD falló (no rompe):", str(e))

    return url

def upload_image_bytes_to_r2_public(image_bytes: bytes, ext: str, prefix: str) -> str:
    content_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")
    return upload_bytes_to_r2_public(image_bytes, ext, prefix, content_type, expect_kind="image")

def upload_video_mp4_to_r2_public(video_bytes: bytes, prefix: str) -> str:
    return upload_bytes_to_r2_public(video_bytes, ".mp4", prefix, "video/mp4", expect_kind="video")

# =========================
# RSS / Images extraction (+ YouTube thumbnail fix)
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
    # Special case: YouTube thumbnail
    if is_youtube_url(page_url):
        vid = extract_youtube_video_id(page_url)
        if vid:
            return youtube_thumbnail_candidates(vid)[:max_images]

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
        # fallback youtube if weird redirect
        if is_youtube_url(page_url):
            vid = extract_youtube_video_id(page_url)
            if vid:
                return youtube_thumbnail_candidates(vid)[:max_images]
        return []

def download_image_bytes(image_url: str) -> Tuple[bytes, str]:
    parsed = urlparse(image_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"URL de imagen inválida: {image_url}")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": image_url,
    }
    r = requests.get(image_url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    ext = _guess_ext_from_content_type(r.headers.get("Content-Type", ""))
    if ext not in [".png", ".webp", ".jpg", ".jpeg"]:
        ext = ".jpg"
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

    # dedupe by link
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for a in raw:
        if a["link"] not in seen:
            seen.add(a["link"])
            deduped.append(a)

    # balance per feed
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
# OpenAI (text)
# =========================

def openai_client():
    if not OpenAI:
        raise RuntimeError("No se pudo importar OpenAI. Revisa requirements.txt (openai).")
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_text(prompt: str) -> str:
    model = env_nonempty("OPENAI_MODEL", OPENAI_MODEL) or "gpt-4.1-mini"
    client = openai_client()

    # Prefer Responses API, fallback to chat.completions
    try:
        resp = client.responses.create(model=model, input=prompt)
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()

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

# =========================
# Threads length guard (<= 500)
# =========================

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
# Threads API (real)
# =========================

def _threads_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def threads_create_container_image(user_id: str, access_token: str, text: str, image_url: str) -> str:
    if not image_url.startswith("http"):
        raise RuntimeError(f"image_url inválido (debe ser https): {image_url}")
    url = f"{THREADS_GRAPH}/{user_id}/threads"
    text = clip_threads_text(text, 500)
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
        r = requests.get(url, headers=_threads_headers(access_token), params={"fields": "status,error_message"}, timeout=HTTP_TIMEOUT)
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
    r = _post_with_retries(url, headers=_threads_headers(access_token), data={"creation_id": container_id}, label="THREADS PUBLISH")
    return r.json()

def threads_publish_text_image(user_id: str, access_token: str, dry_run: bool, text: str, image_url_from_news: str, threads_media_prefix: str) -> Dict[str, Any]:
    if dry_run or (not ENABLE_THREADS_PUBLISH):
        print("[DRY_RUN] Threads disabled or dry_run, no publico.")
        print("[DRY_RUN] Threads post:", clip_threads_text(text, 500))
        print("[DRY_RUN] Image source:", image_url_from_news)
        return {"ok": True, "dry_run": True}

    if not access_token:
        raise RuntimeError("Falta THREADS_USER_ACCESS_TOKEN")

    img_bytes, ext = download_image_bytes(image_url_from_news)
    r2_url = upload_image_bytes_to_r2_public(img_bytes, ext, prefix=threads_media_prefix)
    print("Imagen rehost R2:", r2_url)

    container_id = threads_create_container_image(user_id, access_token, text, r2_url)
    print("Container created:", container_id)

    threads_wait_container(container_id, access_token)
    res = threads_publish(user_id, access_token, container_id)
    print("Threads publish response:", res)

    return {"ok": True, "container": {"id": container_id}, "publish": res, "image_url": r2_url}

# =========================
# Runway (image->video) optional
# =========================

MP4_URL_RE = re.compile(r"https?://[^\s\"']+\.mp4[^\s\"']*", re.IGNORECASE)

def runway_headers() -> Dict[str, str]:
    if not RUNWAY_API_KEY:
        raise RuntimeError("Falta RUNWAY_API_KEY")
    return {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "X-Runway-Version": RUNWAY_VERSION,
        "Content-Type": "application/json",
    }

def runway_create_image_to_video(image_https_url: str, prompt_text: str, seconds: int = 5) -> str:
    url = f"{RUNWAY_BASE}/v1/image_to_video"
    payload = {
        "model": RUNWAY_I2V_MODEL,
        "promptText": prompt_text[:1000],
        "ratio": "720:1280",
        "duration": int(max(2, min(10, seconds))),
        "promptImage": image_https_url,
    }
    r = _post_with_retries(url, headers=runway_headers(), json_body=payload, label="RUNWAY I2V CREATE")
    j = r.json()
    task_id = j.get("id")
    if not task_id:
        raise RuntimeError(f"Runway no devolvió id: {j}")
    return task_id

def runway_get_task(task_id: str) -> Dict[str, Any]:
    url = f"{RUNWAY_BASE}/v1/tasks/{task_id}"
    r = requests.get(url, headers=runway_headers(), timeout=HTTP_TIMEOUT)
    _raise_meta_error(r, "RUNWAY TASK GET")
    return r.json()

def runway_wait_for_mp4(task_id: str, timeout_sec: int = 420) -> str:
    start = time.time()
    last = None
    while time.time() - start < timeout_sec:
        j = runway_get_task(task_id)
        last = j
        status = (j.get("status") or "").upper()
        if status in ("SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED"):
            s = json.dumps(j)
            m = MP4_URL_RE.search(s)
            if m:
                return m.group(0)
            raise RuntimeError(f"Task succeeded but no mp4 found in response: {j}")
        if status in ("FAILED", "ERROR", "CANCELED", "CANCELLED"):
            raise RuntimeError(f"Runway task failed: {j}")
        time.sleep(RUNWAY_POLL_SEC)
    raise TimeoutError(f"Runway task timeout. Last={last}")

def download_runway_mp4_robust(mp4_url: str, task_id: Optional[str] = None) -> bytes:
    """
    Tries hard to download mp4. If it 401s, re-fetches task to get a fresh signed URL and retries.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "video/*,*/*;q=0.8",
        "Referer": "https://app.runwayml.com/",
    }

    last_err = None
    url_to_try = mp4_url

    for attempt in range(RUNWAY_MP4_RETRIES):
        try:
            r = _get_with_retries(url_to_try, headers=headers, timeout=60, label="RUNWAY MP4 DOWNLOAD", retries=1, allow_redirects=True)
            return r.content
        except Exception as e:
            last_err = e

            # If unauthorized, try refreshing URL from task (signed URL may rotate)
            msg = str(e).lower()
            if ("401" in msg or "unauthorized" in msg) and task_id:
                try:
                    j = runway_get_task(task_id)
                    s = json.dumps(j)
                    m = MP4_URL_RE.search(s)
                    if m:
                        url_to_try = m.group(0)
                        print("Runway mp4 URL refreshed:", url_to_try[:120] + ("..." if len(url_to_try) > 120 else ""))
                except Exception:
                    pass

            time.sleep(2.0)

    raise RuntimeError(f"No se pudo descargar mp4 de Runway tras reintentos. Último error: {last_err}")

# =========================
# Reel generator helpers (wrap text, pick music/logo)
# =========================

def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"Falta {label} en repo: {path}")

def wrap_text_lines(text: str, max_chars_per_line: int = 30, max_lines: int = 3) -> str:
    """
    Simple wrapping for ffmpeg drawtext using newline breaks.
    """
    t = (text or "").strip().replace("\n", " ")
    words = t.split()
    if not words:
        return ""

    lines: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + 1 + len(w) <= max_chars_per_line:
            cur = cur + " " + w
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break

    if len(lines) < max_lines and cur:
        lines.append(cur)

    # If overflow, truncate last line
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if lines:
        lines[-1] = lines[-1][:max_chars_per_line].rstrip()
    return "\n".join(lines).strip()

def pick_logo_path(default_logo: str) -> Optional[str]:
    if not default_logo or not os.path.exists(default_logo):
        return None
    if random.random() <= max(0.0, min(1.0, LOGO_PROBABILITY)):
        return default_logo
    return None

def list_mp3_files(search_dir: str) -> List[str]:
    files: List[str] = []
    if search_dir and os.path.isdir(search_dir):
        for root, _, fnames in os.walk(search_dir):
            for f in fnames:
                if f.lower().endswith(".mp3"):
                    files.append(os.path.join(root, f))
    # Also include DEFAULT_ASSET_MUSIC if it's a file and not already in list
    if DEFAULT_ASSET_MUSIC and os.path.isfile(DEFAULT_ASSET_MUSIC) and DEFAULT_ASSET_MUSIC not in files:
        files.append(DEFAULT_ASSET_MUSIC)
    return files

def pick_music_path() -> Optional[str]:
    p = max(0.0, min(1.0, MUSIC_PROBABILITY))
    if random.random() > p:
        print(f"Music selected: NONE (prob={p})")
        return None

    candidates = list_mp3_files(MUSIC_SEARCH_DIR)
    # filter out tiny/bad files
    good = []
    for c in candidates:
        try:
            if os.path.isfile(c) and os.path.getsize(c) > 50_000:  # ~50KB minimum
                good.append(c)
        except Exception:
            continue

    if not good:
        print("Music selected: NONE (mudo)  (no hay mp3 válidos en assets/ o son muy pequeños)")
        return None

    chosen = random.choice(good)
    print("Music selected:", chosen)
    return chosen

def generate_reel_from_image(
    headline: str,
    news_image_path: str,
    logo_path: Optional[str],
    bg_path: str,
    seconds: int,
    music_path: Optional[str] = None,
    cta_text: Optional[str] = None,
) -> bytes:
    """
    Stable template: bg + image + optional logo + wrapped text + optional music.
    """
    _require_file(bg_path, "ASSET_BG")
    if not os.path.exists(news_image_path):
        raise RuntimeError(f"Falta news image local: {news_image_path}")
    if not os.path.exists(FONT_BOLD):
        raise RuntimeError(f"Falta FONT_BOLD en runner: {FONT_BOLD}")

    headline_wrapped = wrap_text_lines((headline or "")[:220], max_chars_per_line=30, max_lines=3)
    cta = (cta_text or "Sigue para más").strip()

    music_ok = bool(music_path) and os.path.exists(music_path)
    logo_ok = bool(logo_path) and os.path.exists(logo_path) if logo_path else False

    # Choose font sizes based on number of lines
    n_lines = max(1, headline_wrapped.count("\n") + 1)
    if n_lines == 1:
        title_size = 54
    elif n_lines == 2:
        title_size = 50
    else:
        title_size = 44

    with tempfile.TemporaryDirectory() as td:
        out_mp4 = os.path.join(td, "reel.mp4")
        title_txt = os.path.join(td, "title.txt")
        cta_txt = os.path.join(td, "cta.txt")

        with open(title_txt, "w", encoding="utf-8") as f:
            f.write(headline_wrapped)
        with open(cta_txt, "w", encoding="utf-8") as f:
            f.write(cta)

        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={REEL_W}x{REEL_H}:r=30:d={int(seconds)}",
            "-i", bg_path,
            "-i", news_image_path,
        ]

        # optional logo
        if logo_ok:
            cmd += ["-i", logo_path]  # index 3 when present

        # optional music
        if music_ok:
            cmd += ["-i", music_path]  # last input

        # filter graph
        # Inputs:
        # 0: black base
        # 1: bg
        # 2: news image
        # 3: logo (optional)
        # audio: last (optional)

        vf_parts = []
        vf_parts.append(f"[1:v]scale={REEL_W}:{REEL_H},format=rgba[bg];")
        vf_parts.append(f"[0:v][bg]overlay=0:0:format=auto[v1];")
        vf_parts.append(f"[2:v]scale={REEL_W-120}:-1,format=rgba[news];")
        vf_parts.append(f"[v1][news]overlay=(W-w)/2:520:format=auto[v2];")

        if logo_ok:
            vf_parts.append(f"[3:v]scale=700:-1,format=rgba[logo];")
            vf_parts.append(f"[v2][logo]overlay=(W-w)/2:170:format=auto[v3];")
            base_label = "[v3]"
        else:
            base_label = "[v2]"

        vf_parts.append(
            f"{base_label}"
            f"drawtext=fontfile={FONT_BOLD}:textfile={title_txt}:x=60:y=1280:"
            f"fontsize={title_size}:line_spacing=10:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=24,"
            f"drawtext=fontfile={FONT_BOLD}:textfile={cta_txt}:x=60:y=1560:"
            f"fontsize=44:fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=18"
            f"[vout]"
        )

        vf = "".join(vf_parts)

        cmd += ["-filter_complex", vf, "-map", "[vout]"]

        # audio mapping:
        if music_ok:
            # music is last input
            audio_index = 3 if not logo_ok else 4
            cmd += ["-map", f"{audio_index}:a", "-filter:a", "volume=0.35", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]

        cmd += [
            "-t", str(int(seconds)),
            "-r", "30",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-shortest",
            out_mp4
        ]

        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg falló:\nSTDERR:\n{(p.stderr or '')[:4000]}")

        with open(out_mp4, "rb") as f:
            return f.read()

def generate_reel_from_video_bg(
    headline: str,
    bg_video_path: str,
    logo_path: Optional[str],
    seconds: int,
    music_path: Optional[str] = None,
    cta_text: Optional[str] = None,
) -> bytes:
    """
    Use a vertical bg video (e.g. Runway i2v output) as reel background + overlays.
    """
    if not os.path.exists(bg_video_path):
        raise RuntimeError(f"Falta bg video local: {bg_video_path}")
    if not os.path.exists(FONT_BOLD):
        raise RuntimeError(f"Falta FONT_BOLD en runner: {FONT_BOLD}")

    headline_wrapped = wrap_text_lines((headline or "")[:220], max_chars_per_line=30, max_lines=3)
    cta = (cta_text or "Sigue para más").strip()

    music_ok = bool(music_path) and os.path.exists(music_path)
    logo_ok = bool(logo_path) and os.path.exists(logo_path) if logo_path else False

    n_lines = max(1, headline_wrapped.count("\n") + 1)
    if n_lines == 1:
        title_size = 56
    elif n_lines == 2:
        title_size = 50
    else:
        title_size = 44

    with tempfile.TemporaryDirectory() as td:
        out_mp4 = os.path.join(td, "reel.mp4")
        title_txt = os.path.join(td, "title.txt")
        cta_txt = os.path.join(td, "cta.txt")

        with open(title_txt, "w", encoding="utf-8") as f:
            f.write(headline_wrapped)
        with open(cta_txt, "w", encoding="utf-8") as f:
            f.write(cta)

        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", bg_video_path,
        ]
        if logo_ok:
            cmd += ["-i", logo_path]
        if music_ok:
            cmd += ["-i", music_path]

        vf_parts = []
        vf_parts.append(
            f"[0:v]scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
            f"crop={REEL_W}:{REEL_H},fps=30,format=rgba[v0];"
        )
        if logo_ok:
            vf_parts.append(f"[1:v]scale=520:-1,format=rgba[logo];")
            vf_parts.append(f"[v0][logo]overlay=40:60:format=auto[v1];")
            base_label = "[v1]"
        else:
            base_label = "[v0]"

        vf_parts.append(
            f"{base_label}"
            f"drawtext=fontfile={FONT_BOLD}:textfile={title_txt}:x=60:y=1280:"
            f"fontsize={title_size}:line_spacing=10:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=24,"
            f"drawtext=fontfile={FONT_BOLD}:textfile={cta_txt}:x=60:y=1560:"
            f"fontsize=44:fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=18"
            f"[vout]"
        )
        vf = "".join(vf_parts)

        cmd += ["-filter_complex", vf, "-map", "[vout]"]

        if music_ok:
            # if logo exists, music is input 2 else 1
            audio_index = 2 if logo_ok else 1
            cmd += ["-map", f"{audio_index}:a", "-filter:a", "volume=0.35", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]

        cmd += [
            "-t", str(int(seconds)),
            "-r", "30",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-shortest",
            out_mp4
        ]

        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=480, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg (video bg) falló:\nSTDERR:\n{(p.stderr or '')[:4000]}")

        with open(out_mp4, "rb") as f:
            return f.read()

# =========================
# Instagram Graph API publish
# =========================

def ig_api_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.post(url, data=data, timeout=HTTP_TIMEOUT)
    _raise_meta_error(r, f"IG POST {path}")
    return r.json()

def ig_api_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    _raise_meta_error(r, f"IG GET {path}")
    return r.json()

def ig_wait_container(creation_id: str, access_token: str, timeout_sec: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        j = ig_api_get(f"{creation_id}", {"fields": "status_code", "access_token": access_token})
        status = (j.get("status_code") or "").upper()
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {j}")
        time.sleep(3)
    raise TimeoutError(f"IG container not ready after {timeout_sec}s")

def ig_publish_reel(video_url: str, caption: str) -> Dict[str, Any]:
    if not (IG_USER_ID and IG_ACCESS_TOKEN):
        raise RuntimeError("Faltan IG_USER_ID o IG_ACCESS_TOKEN")

    print("IG publish: creando container...")
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

    print("IG publish: esperando container...", creation_id)
    ig_wait_container(creation_id, IG_ACCESS_TOKEN, timeout_sec=900)

    print("IG publish: publicando...")
    res = ig_api_post(f"{IG_USER_ID}/media_publish", {"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN})
    return res

# =========================
# State
# =========================

def load_threads_state(state_key: str) -> Dict[str, Any]:
    state = load_from_r2_json(state_key)
    if not state:
        state = {"posted_items": {}, "posted_links": [], "last_posted_at": None}
    state.setdefault("posted_items", {})
    state.setdefault("posted_links", [])
    state.setdefault("last_posted_at", None)
    return state

def save_threads_state(state_key: str, state: Dict[str, Any]) -> None:
    save_to_r2_json(state_key, state)

def mark_posted(state: Dict[str, Any], link: str) -> None:
    pi = state["posted_items"].get(link, {"times": 0, "last_posted_at": None})
    pi["times"] = int(pi.get("times", 0)) + 1
    pi["last_posted_at"] = iso_now()
    state["posted_items"][link] = pi
    if link not in state["posted_links"]:
        state["posted_links"].append(link)
    state["posted_links"] = state["posted_links"][-400:]
    state["last_posted_at"] = iso_now()

def is_new_allowed(state: Dict[str, Any], link: str) -> bool:
    return link not in state["posted_items"]

def repost_eligible(state: Dict[str, Any], link: str, repost_max_times: int, repost_window_days: int) -> bool:
    pi = state["posted_items"].get(link)
    if not pi:
        return False
    times = int(pi.get("times", 0))
    if times >= repost_max_times:
        return False
    last = pi.get("last_posted_at")
    if not last:
        return True
    return days_since(last) >= repost_window_days

def pick_item(articles: List[Dict[str, Any]], state: Dict[str, Any], repost_enable: bool, repost_max_times: int, repost_window_days: int) -> Tuple[Optional[Dict[str, Any]], str]:
    for a in articles:
        if a.get("link") and is_new_allowed(state, a["link"]):
            return a, "new"
    if repost_enable:
        for a in articles:
            if a.get("link") and a["link"] in state["posted_items"] and repost_eligible(state, a["link"], repost_max_times, repost_window_days):
                return a, "repost"
    return None, "none"

# =========================
# Accounts loading
# =========================

def load_accounts() -> List[Dict[str, Any]]:
    try:
        with open("accounts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "accounts" in data and isinstance(data["accounts"], list):
            return data["accounts"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

# =========================
# Main per account run
# =========================

def run_account(cfg: Dict[str, Any]) -> Dict[str, Any]:
    account_id = cfg.get("account_id", "unknown")
    print(f"\n===== RUN ACCOUNT: {account_id} =====")

    rss_feeds = cfg.get("rss_feeds") or []
    max_per_feed = int(cfg.get("max_per_feed", 3))
    shuffle = bool(cfg.get("shuffle_articles", True))
    max_ai_items = int(cfg.get("max_ai_items", 15))

    assets_cfg = cfg.get("assets", {}) or {}
    asset_bg = assets_cfg.get("bg") or DEFAULT_ASSET_BG
    asset_logo_default = assets_cfg.get("logo") or DEFAULT_ASSET_LOGO
    asset_music_hint = assets_cfg.get("music") or DEFAULT_ASSET_MUSIC
    cta_text = assets_cfg.get("cta") or "Sigue para más"

    threads_cfg = cfg.get("threads", {})
    threads_user_id = threads_cfg.get("user_id", THREADS_USER_ID or "me")
    state_key = threads_cfg.get("state_key", f"accounts/{account_id}/threads_state.json")
    auto_post_limit = int(threads_cfg.get("auto_post_limit", 1))
    acct_dry_run = bool(threads_cfg.get("dry_run", False)) or DRY_RUN
    repost_enable = bool(threads_cfg.get("repost_enable", True))
    repost_max_times = int(threads_cfg.get("repost_max_times", 3))
    repost_window_days = int(threads_cfg.get("repost_window_days", 7))

    r2_cfg = cfg.get("r2", {})
    threads_media_prefix = (r2_cfg.get("threads_media_prefix") or f"threads_media/{account_id}").strip().strip("/")
    reels_prefix = (r2_cfg.get("reels_prefix") or f"ugc/reels/{account_id}").strip().strip("/")

    if auto_post_limit <= 0 or not rss_feeds:
        print(f"Cuenta {account_id} está apagada (auto_post_limit=0 o rss_feeds vacío). Saltando ✅")
        return {"generated_at": iso_now(), "account_id": account_id, "skipped": True, "reason": "disabled_or_no_feeds"}

    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles(rss_feeds, max_per_feed=max_per_feed, shuffle=shuffle)
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={max_per_feed}, SHUFFLE={shuffle})")

    processed = list(articles[:max_ai_items])
    state = load_threads_state(state_key)
    print("STATE posted_items:", len(state.get("posted_items", {})))

    posted_count = 0
    results: List[Dict[str, Any]] = []

    while posted_count < auto_post_limit:
        item, mode = pick_item(processed, state, repost_enable, repost_max_times, repost_window_days)
        if not item:
            print("No hay item nuevo ni repost elegible.")
            break

        link = item["link"]
        label = "NUEVO" if mode == "new" else "REPOST"
        print(f"Seleccionado ({label}): {link}")

        # text
        try:
            threads_text = build_threads_text(item, mode=mode)
        except Exception as e:
            print("OpenAI falló generando texto (se omite item):", str(e))
            processed = [x for x in processed if x.get("link") != link]
            continue

        # image candidates
        img_candidates = extract_best_images(link, max_images=5)
        if not img_candidates:
            print("No se encontró imagen. Se omite.")
            processed = [x for x in processed if x.get("link") != link]
            continue

        chosen_img = None
        for u in img_candidates:
            try:
                _ = download_image_bytes(u)
                chosen_img = u
                break
            except Exception:
                continue

        if not chosen_img:
            print("Todas las imágenes del artículo fallaron. Se omite.")
            processed = [x for x in processed if x.get("link") != link]
            continue

        # Threads publish (or dry-run)
        threads_res = None
        try:
            threads_res = threads_publish_text_image(
                user_id=threads_user_id,
                access_token=THREADS_USER_ACCESS_TOKEN or "",
                dry_run=acct_dry_run,
                text=threads_text,
                image_url_from_news=chosen_img,
                threads_media_prefix=threads_media_prefix,
            )
        except Exception as e:
            print("Threads falló (no rompe todo):", str(e))
            threads_res = {"ok": False, "error": str(e)}

        # Reel generation
        reel_url = None
        if ENABLE_REELS:
            try:
                # download chosen image locally
                img_bytes, img_ext = download_image_bytes(chosen_img)

                # rehost to r2 for runway input
                img_r2_url = upload_image_bytes_to_r2_public(img_bytes, img_ext, prefix=threads_media_prefix)
                print("Imagen rehost R2:", img_r2_url)

                # choose music + logo randomly
                chosen_music = pick_music_path()
                chosen_logo = pick_logo_path(asset_logo_default)

                with tempfile.TemporaryDirectory() as td:
                    local_img = os.path.join(td, f"news{img_ext}")
                    with open(local_img, "wb") as f:
                        f.write(img_bytes)

                    use_runway = RUNWAY_ENABLED and bool(RUNWAY_API_KEY)
                    if use_runway:
                        try:
                            runway_prompt = (
                                "Cinematic esports/gaming animation, neon, glitch, HUD overlays, "
                                "dynamic camera movement, high energy, viral reel background. "
                                f"Based on this news image and headline: {item.get('title','')}"
                            )
                            task_id = runway_create_image_to_video(img_r2_url, runway_prompt, seconds=RUNWAY_I2V_SECONDS)
                            print("Runway task created:", task_id)

                            mp4_url = runway_wait_for_mp4(task_id, timeout_sec=RUNWAY_TIMEOUT)
                            print("Runway mp4 URL:", mp4_url)

                            mp4_bytes = download_runway_mp4_robust(mp4_url, task_id=task_id)
                            bg_vid = os.path.join(td, "bg.mp4")
                            with open(bg_vid, "wb") as f:
                                f.write(mp4_bytes)

                            reel_bytes = generate_reel_from_video_bg(
                                headline=item.get("title", "")[:220],
                                bg_video_path=bg_vid,
                                logo_path=chosen_logo,
                                seconds=REEL_SECONDS,
                                music_path=chosen_music,
                                cta_text=cta_text,
                            )
                        except Exception as e:
                            print("Runway falló (fallback a reel normal):", str(e))
                            reel_bytes = generate_reel_from_image(
                                headline=item.get("title", "")[:220],
                                news_image_path=local_img,
                                logo_path=chosen_logo,
                                bg_path=asset_bg,
                                seconds=REEL_SECONDS,
                                music_path=chosen_music,
                                cta_text=cta_text,
                            )
                    else:
                        reel_bytes = generate_reel_from_image(
                            headline=item.get("title", "")[:220],
                            news_image_path=local_img,
                            logo_path=chosen_logo,
                            bg_path=asset_bg,
                            seconds=REEL_SECONDS,
                            music_path=chosen_music,
                            cta_text=cta_text,
                        )

                reel_url = upload_video_mp4_to_r2_public(reel_bytes, prefix=reels_prefix)
                print("Reel URL R2:", reel_url)
            except Exception as e:
                print("Reel generation falló (no rompe):", str(e))

        # IG publish (optional)
        ig_res = None
        ig_kind = None
        if reel_url:
            ig_kind = "reel"
            if ENABLE_IG_PUBLISH and (not acct_dry_run):
                try:
                    ig_caption = build_instagram_caption(item, link)
                    ig_res = ig_publish_reel(video_url=reel_url, caption=ig_caption)
                    print("IG publish OK:", ig_res)
                except Exception as e:
                    print("IG publish falló (no rompe):", str(e))
                    ig_res = {"ok": False, "error": str(e), "video_url": reel_url}
            else:
                print("[DRY_RUN] IG disabled or dry_run, no publico.")
                ig_res = {"video_url": reel_url, "published": False}
        else:
            if ENABLE_IG_PUBLISH and (not acct_dry_run):
                print("IG: no hay reel_url, entonces NO se publica (esto pasa si falló la generación del reel).")

        # mark posted in state if Threads actually published
        if threads_res and threads_res.get("ok") and not threads_res.get("dry_run"):
            mark_posted(state, link)
            save_threads_state(state_key, state)
            posted_count += 1
        elif threads_res and threads_res.get("dry_run"):
            posted_count += 1

        results.append(
            {
                "link": link,
                "mode": mode,
                "threads": threads_res,
                "ig": ig_res,
                "ig_kind": ig_kind,
                "dry_run": acct_dry_run,
            }
        )
        break

    run_payload = {
        "generated_at": iso_now(),
        "account_id": account_id,
        "settings": {
            "enable_reels": ENABLE_REELS,
            "reel_seconds": REEL_SECONDS,
            "enable_ig_publish": ENABLE_IG_PUBLISH,
            "enable_threads_publish": ENABLE_THREADS_PUBLISH,
            "dry_run": DRY_RUN,
            "graph_version": f"v{GRAPH_VERSION}",
            "runway_enabled": RUNWAY_ENABLED and bool(RUNWAY_API_KEY),
            "music_probability": MUSIC_PROBABILITY,
            "logo_probability": LOGO_PROBABILITY,
        },
        "result": {"posted_count": posted_count, "results": results},
    }
    return run_payload

def save_run_payload(account_id: str, payload: Dict[str, Any]) -> str:
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    key = f"accounts/{account_id}/runs/editorial_run_{run_id}.json"
    save_to_r2_json(key, payload)
    print("Archivo guardado en R2:", key)
    return key

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    accounts = load_accounts()
    if not accounts:
        raise RuntimeError("No se encontraron cuentas. Falta accounts.json o está vacío.")

    all_results = []
    for cfg in accounts:
        account_id = cfg.get("account_id", "unknown")
        payload = run_account(cfg)
        run_key = save_run_payload(account_id, payload)
        all_results.append({"account_id": account_id, "run_key": run_key, "payload": payload})
        print("RUN COMPLETED:", account_id)

    print("\n===== SUMMARY =====")
    print(json.dumps({"runs": all_results}, ensure_ascii=False, indent=2))
