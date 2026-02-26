import os
import re
import json
import time
import hashlib
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
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


print("RUNNING MEDIA ENGINE (Threads REAL + IG Queue + REEL AUTO + IG PUBLISH + Multi-account via accounts.json)")

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

R2_PUBLIC_BASE_URL = env_nonempty(
    "R2_PUBLIC_BASE_URL",
    "https://pub-8937244ee725495691514507bb8f431e.r2.dev"
).rstrip("/")

THREADS_GRAPH = env_nonempty("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")
THREADS_USER_ACCESS_TOKEN = env_nonempty("THREADS_USER_ACCESS_TOKEN")

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)
POST_RETRY_MAX = env_int("POST_RETRY_MAX", 2)
POST_RETRY_SLEEP = env_float("POST_RETRY_SLEEP", 2.0)

CONTAINER_WAIT_TIMEOUT = env_int("CONTAINER_WAIT_TIMEOUT", 120)
CONTAINER_POLL_INTERVAL = env_float("CONTAINER_POLL_INTERVAL", 2.0)

IG_CAROUSEL_MAX_IMAGES = env_int("IG_CAROUSEL_MAX_IMAGES", 5)

VERIFY_NEWS = env_bool("VERIFY_NEWS", False)
ENABLE_TRENDS = env_bool("ENABLE_TRENDS", False)

# Reels generation
ENABLE_REELS = env_bool("ENABLE_REELS", True)
REEL_SECONDS = env_int("REEL_SECONDS", 15)
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)

DEFAULT_ASSET_BG = env_nonempty("ASSET_BG", "assets/bg.jpg")
DEFAULT_ASSET_LOGO = env_nonempty("ASSET_LOGO", "assets/logo.png")
DEFAULT_ASSET_MUSIC = env_nonempty("ASSET_MUSIC", "assets/music.mp3")  # opcional

FONT_BOLD = env_nonempty("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Instagram publish (Graph API)
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", False)
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
IG_USER_ID = env_nonempty("IG_USER_ID")
GRAPH_VERSION = env_nonempty("GRAPH_VERSION", "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"


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


# =========================
# HTTP helpers
# =========================

def _threads_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

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
    if "mp4" in ct:
        return ".mp4"
    return ".bin"

def upload_bytes_to_r2_public(file_bytes: bytes, ext: str, prefix: str, content_type: str, expect_kind: str = "any") -> str:
    base = env_nonempty("R2_PUBLIC_BASE_URL", R2_PUBLIC_BASE_URL).rstrip("/")
    if not base.startswith("http"):
        raise RuntimeError("R2_PUBLIC_BASE_URL inválido. Debe empezar por https://")

    s3 = r2_client()
    h = hashlib.sha1(file_bytes).hexdigest()[:16]
    prefix = (prefix or "").strip().strip("/")
    key = f"{prefix}/{h}{ext}" if prefix else f"{h}{ext}"

    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_bytes, ContentType=content_type)
    url = f"{base}/{key}"

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

    if not url.startswith("http"):
        raise RuntimeError(f"R2 URL inválido generado: {url}")

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
# RSS / Images extraction
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
    if dry_run:
        print("[DRY_RUN] Threads post:", clip_threads_text(text, 500))
        print("[DRY_RUN] Image source:", image_url_from_news)
        return {"ok": True, "dry_run": True}

    if not access_token:
        raise RuntimeError("Falta THREADS_USER_ACCESS_TOKEN")

    img_bytes, ext = download_image_bytes(image_url_from_news)
    r2_url = upload_image_bytes_to_r2_public(img_bytes, ext, prefix=threads_media_prefix)
    print("IMAGE re-hosted on R2:", r2_url)

    container_id = threads_create_container_image(user_id, access_token, text, r2_url)
    print("Container created:", container_id)

    threads_wait_container(container_id, access_token)
    res = threads_publish(user_id, access_token, container_id)
    print("Threads publish response:", res)

    return {"ok": True, "container": {"id": container_id}, "publish": res, "image_url": r2_url}


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
# IG queue helpers
# =========================

def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def choose_ig_format(has_video: bool, image_count: int) -> str:
    if has_video:
        return "reel"
    if image_count >= 2:
        return "carousel"
    return "image"

def save_ig_queue_item(prefix: str, payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    h = _short_hash(s + iso_now())
    key = f"{prefix}/{now_utc().strftime('%Y%m%d_%H%M%S')}_{h}.json"
    save_to_r2_json(key, payload)
    return key


# =========================
# REEL generator (stable lavfi base)
# =========================

def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"Falta {label} en repo: {path}")

def generate_reel_mp4_bytes(headline: str, news_image_path: str, logo_path: str, bg_path: str, seconds: int, music_path: Optional[str] = None, cta_text: Optional[str] = None) -> bytes:
    _require_file(bg_path, "ASSET_BG")
    _require_file(logo_path, "ASSET_LOGO")
    if not os.path.exists(news_image_path):
        raise RuntimeError(f"Falta news image local: {news_image_path}")
    if not os.path.exists(FONT_BOLD):
        raise RuntimeError(f"Falta FONT_BOLD en runner: {FONT_BOLD}")

    headline_clean = (headline or "").strip().replace("\n", " ")[:140]
    cta = (cta_text or "Sigue para más").strip()
    music_ok = bool(music_path) and os.path.exists(music_path)

    with tempfile.TemporaryDirectory() as td:
        out_mp4 = os.path.join(td, "reel.mp4")
        title_txt = os.path.join(td, "title.txt")
        cta_txt = os.path.join(td, "cta.txt")

        with open(title_txt, "w", encoding="utf-8") as f:
            f.write(headline_clean)
        with open(cta_txt, "w", encoding="utf-8") as f:
            f.write(cta)

        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={REEL_W}x{REEL_H}:r=30:d={int(seconds)}",
            "-i", bg_path,
            "-i", news_image_path,
            "-i", logo_path,
        ]
        if music_ok:
            cmd += ["-i", music_path]  # index 4

        vf = (
            f"[1:v]scale={REEL_W}:{REEL_H},format=rgba[bg];"
            f"[0:v][bg]overlay=0:0:format=auto[v1];"
            f"[2:v]scale={REEL_W-120}:-1,format=rgba[news];"
            f"[v1][news]overlay=(W-w)/2:520:format=auto[v2];"
            f"[3:v]scale=700:-1,format=rgba[logo];"
            f"[v2][logo]overlay=(W-w)/2:170:format=auto[v3];"
            f"[v3]"
            f"drawtext=fontfile={FONT_BOLD}:textfile={title_txt}:x=60:y=1320:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=24,"
            f"drawtext=fontfile={FONT_BOLD}:textfile={cta_txt}:x=60:y=1540:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=18"
            f"[vout]"
        )

        cmd += ["-filter_complex", vf, "-map", "[vout]"]

        if music_ok:
            cmd += ["-map", "4:a", "-filter:a", "volume=0.15", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]

        cmd += ["-t", str(seconds), "-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", out_mp4]

        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180, check=False)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg se demoró demasiado y se cortó por timeout (180s).")

        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg falló:\nSTDERR:\n{(p.stderr or '')[:4000]}")

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

def ig_wait_container(creation_id: str, access_token: str, timeout_sec: int = 300) -> None:
    url = f"{GRAPH_BASE}/{creation_id}"
    start = time.time()
    while time.time() - start < timeout_sec:
        r = requests.get(url, params={"fields": "status_code", "access_token": access_token}, timeout=HTTP_TIMEOUT)
        _raise_meta_error(r, "IG CONTAINER STATUS")
        j = r.json()
        status = (j.get("status_code") or "").upper()
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {j}")
        time.sleep(3)
    raise TimeoutError(f"IG container not ready after {timeout_sec}s")

def ig_publish_media(ig_user_id: str, access_token: str, creation_id: str) -> Dict[str, Any]:
    return ig_api_post(f"{ig_user_id}/media_publish", {"creation_id": creation_id, "access_token": access_token})

def ig_publish_image(ig_user_id: str, access_token: str, image_url: str, caption: str) -> Dict[str, Any]:
    j = ig_api_post(f"{ig_user_id}/media", {"image_url": image_url, "caption": caption, "access_token": access_token})
    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG media create no devolvió id: {j}")
    return ig_publish_media(ig_user_id, access_token, creation_id)

def ig_publish_carousel(ig_user_id: str, access_token: str, image_urls: List[str], caption: str) -> Dict[str, Any]:
    children_ids: List[str] = []
    for u in image_urls:
        j = ig_api_post(f"{ig_user_id}/media", {"image_url": u, "is_carousel_item": "true", "access_token": access_token})
        cid = j.get("id")
        if not cid:
            raise RuntimeError(f"IG carousel item failed: {j}")
        children_ids.append(cid)

    j2 = ig_api_post(
        f"{ig_user_id}/media",
        {"media_type": "CAROUSEL", "children": ",".join(children_ids), "caption": caption, "access_token": access_token},
    )
    creation_id = j2.get("id")
    if not creation_id:
        raise RuntimeError(f"IG carousel container failed: {j2}")
    return ig_publish_media(ig_user_id, access_token, creation_id)

def ig_publish_reel(ig_user_id: str, access_token: str, video_url: str, caption: str) -> Dict[str, Any]:
    # For Reels: media_type=REELS + video_url
    j = ig_api_post(
        f"{ig_user_id}/media",
        {"media_type": "REELS", "video_url": video_url, "caption": caption, "access_token": access_token},
    )
    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG reels create failed: {j}")
    # Some accounts need wait before publish
    try:
        ig_wait_container(creation_id, access_token, timeout_sec=420)
    except Exception as e:
        print("IG wait (aviso):", str(e))
    return ig_publish_media(ig_user_id, access_token, creation_id)

def maybe_publish_to_instagram(ig_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not ENABLE_IG_PUBLISH:
        return None
    if not (IG_ACCESS_TOKEN and IG_USER_ID):
        print("ENABLE_IG_PUBLISH=true pero faltan IG_ACCESS_TOKEN o IG_USER_ID. Saltando publish IG.")
        return None

    fmt = ig_payload.get("format")
    caption = ig_payload.get("caption") or ""
    assets = ig_payload.get("assets") or {}
    images = (assets.get("images") or [])[:IG_CAROUSEL_MAX_IMAGES]
    video_url = assets.get("reel_video_url")

    print(f"IG PUBLISH: intentando publicar formato={fmt}")

    try:
        if fmt == "reel" and video_url:
            res = ig_publish_reel(IG_USER_ID, IG_ACCESS_TOKEN, video_url, caption)
            print("IG PUBLISH OK (reel):", res)
            return {"ok": True, "format": "reel", "res": res}

        if fmt == "carousel" and len(images) >= 2:
            res = ig_publish_carousel(IG_USER_ID, IG_ACCESS_TOKEN, images, caption)
            print("IG PUBLISH OK (carousel):", res)
            return {"ok": True, "format": "carousel", "res": res}

        if fmt == "image" and len(images) >= 1:
            res = ig_publish_image(IG_USER_ID, IG_ACCESS_TOKEN, images[0], caption)
            print("IG PUBLISH OK (image):", res)
            return {"ok": True, "format": "image", "res": res}

        # fallback
        if video_url:
            res = ig_publish_reel(IG_USER_ID, IG_ACCESS_TOKEN, video_url, caption)
            print("IG PUBLISH OK (fallback reel):", res)
            return {"ok": True, "format": "reel", "res": res}

        if images:
            res = ig_publish_image(IG_USER_ID, IG_ACCESS_TOKEN, images[0], caption)
            print("IG PUBLISH OK (fallback image):", res)
            return {"ok": True, "format": "image", "res": res}

        print("IG PUBLISH: no hay assets válidos para publicar.")
        return {"ok": False, "reason": "no_assets"}

    except Exception as e:
        print("IG PUBLISH FALLÓ (no rompe el run):", str(e))
        return {"ok": False, "error": str(e)}


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
    asset_logo = assets_cfg.get("logo") or DEFAULT_ASSET_LOGO
    asset_music = assets_cfg.get("music") or DEFAULT_ASSET_MUSIC
    cta_text = assets_cfg.get("cta") or "Sigue para más"

    threads_cfg = cfg.get("threads", {})
    threads_user_id = threads_cfg.get("user_id", "me")
    state_key = threads_cfg.get("state_key", f"accounts/{account_id}/threads_state.json")
    auto_post = bool(threads_cfg.get("auto_post", True))
    auto_post_limit = int(threads_cfg.get("auto_post_limit", 1))
    dry_run = bool(threads_cfg.get("dry_run", False))
    repost_enable = bool(threads_cfg.get("repost_enable", True))
    repost_max_times = int(threads_cfg.get("repost_max_times", 3))
    repost_window_days = int(threads_cfg.get("repost_window_days", 7))

    r2_cfg = cfg.get("r2", {})
    threads_media_prefix = r2_cfg.get("threads_media_prefix", f"threads_media/{account_id}").strip().strip("/")
    ig_queue_prefix = r2_cfg.get("ig_queue_prefix", f"ugc/ig_queue/{account_id}").strip().strip("/")
    reels_prefix = r2_cfg.get("reels_prefix", f"ugc/reels/{account_id}").strip().strip("/")

    if auto_post_limit <= 0 or not rss_feeds:
        print(f"Cuenta {account_id} está apagada (auto_post_limit=0 o rss_feeds vacío). Saltando ✅")
        return {"generated_at": iso_now(), "account_id": account_id, "skipped": True, "reason": "disabled_or_no_feeds"}

    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles(rss_feeds, max_per_feed=max_per_feed, shuffle=shuffle)
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={max_per_feed}, SHUFFLE={shuffle})")

    processed = []
    for a in articles[:max_ai_items]:
        processed.append(a)
        time.sleep(0.03)

    state = load_threads_state(state_key)
    print("STATE posted_items:", len(state.get("posted_items", {})))

    posted_count = 0
    results = []

    while posted_count < auto_post_limit:
        item, mode = pick_item(processed, state, repost_enable, repost_max_times, repost_window_days)
        if not item:
            print("No hay item nuevo ni repost elegible.")
            break

        link = item["link"]
        label = "NUEVO" if mode == "new" else "REPOST"
        print(f"Seleccionado ({label}): {link}")

        # Genera texto
        try:
            text = build_threads_text(item, mode=mode)
        except Exception as e:
            print("OpenAI falló generando texto (se omite item):", str(e))
            processed = [x for x in processed if x.get("link") != link]
            continue

        # Buscar imagen
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

        if not auto_post:
            print("THREADS_AUTO_POST desactivado. (No se publica)")
            results.append({"link": link, "mode": mode, "posted": False, "reason": "auto_post_off"})
            break

        # Publicar en Threads
        print(f"Publicando en Threads ({label})...")
        try:
            threads_res = threads_publish_text_image(
                user_id=threads_user_id,
                access_token=THREADS_USER_ACCESS_TOKEN,
                dry_run=dry_run,
                text=text,
                image_url_from_news=chosen_img,
                threads_media_prefix=threads_media_prefix,
            )

            if not dry_run:
                mark_posted(state, link)
                save_threads_state(state_key, state)

            posted_count += 1
            results.append({"link": link, "mode": mode, "posted": True, "threads": threads_res})
            print("Auto-post Threads: OK ✅")

            # IG queue + reels + publish
            try:
                candidates = extract_best_images(link, max_images=IG_CAROUSEL_MAX_IMAGES)

                r2_images: List[str] = []
                for img_url in candidates:
                    try:
                        b, ext = download_image_bytes(img_url)
                        r2_img = upload_image_bytes_to_r2_public(b, ext, prefix=threads_media_prefix)
                        if r2_img not in r2_images:
                            r2_images.append(r2_img)
                    except Exception:
                        continue
                    if len(r2_images) >= IG_CAROUSEL_MAX_IMAGES:
                        break

                threads_img = threads_res.get("image_url") if isinstance(threads_res, dict) else None
                if not r2_images and threads_img:
                    r2_images = [threads_img]

                has_video = False
                reel_video_url = None

                if ENABLE_REELS and r2_images:
                    print(f"Generando REEL automático ({REEL_SECONDS}s)...")
                    try:
                        src_img_for_reel = r2_images[0]
                        rr = requests.get(src_img_for_reel, timeout=30)
                        rr.raise_for_status()
                        img_bytes = rr.content

                        with tempfile.TemporaryDirectory() as td:
                            news_img_path = os.path.join(td, "news.jpg")
                            with open(news_img_path, "wb") as f:
                                f.write(img_bytes)

                            reel_bytes = generate_reel_mp4_bytes(
                                headline=(item.get("title") or "Update esports"),
                                news_image_path=news_img_path,
                                logo_path=asset_logo,
                                bg_path=asset_bg,
                                seconds=REEL_SECONDS,
                                music_path=asset_music,
                                cta_text=cta_text
                            )

                        reel_video_url = upload_video_mp4_to_r2_public(reel_bytes, prefix=reels_prefix)
                        has_video = True
                        print("REEL subido a R2:", reel_video_url)
                    except Exception as e:
                        print("REEL: falló (no rompe el run):", str(e))

                ig_format = choose_ig_format(has_video=has_video, image_count=len(r2_images))
                ig_caption = build_instagram_caption(item, link=link)

                ig_payload = {
                    "created_at": iso_now(),
                    "account_id": account_id,
                    "source": {"link": link, "feed": item.get("feed"), "mode": mode, "title": item.get("title")},
                    "format": ig_format,
                    "caption": ig_caption,
                    "assets": {"images": r2_images[:IG_CAROUSEL_MAX_IMAGES], "reel_video_url": reel_video_url},
                    "threads": {
                        "publish_id": (threads_res.get("publish") or {}).get("id") if isinstance(threads_res, dict) else None,
                        "image_url": threads_img,
                    },
                }

                ig_key = save_ig_queue_item(ig_queue_prefix, ig_payload)
                print("IG queue guardado en R2:", ig_key)

                pub_res = maybe_publish_to_instagram(ig_payload)
                if pub_res is not None:
                    print("IG publish result:", pub_res)

            except Exception as e:
                print("IG pipeline: falló (no rompe el run):", str(e))

        except Exception as e:
            print("Auto-post Threads: FALLÓ ❌")
            print("ERROR:", str(e))
            results.append({"link": link, "mode": mode, "posted": False, "error": str(e)})
            processed = [x for x in processed if x.get("link") != link]
            continue

        processed = [x for x in processed if x.get("link") != link]

    run_payload = {
        "generated_at": iso_now(),
        "account_id": account_id,
        "mix": {"shuffle": shuffle, "max_per_feed": max_per_feed, "max_ai_items": max_ai_items},
        "settings": {
            "verify_news": VERIFY_NEWS,
            "enable_trends": ENABLE_TRENDS,
            "enable_reels": ENABLE_REELS,
            "reel_seconds": REEL_SECONDS,
            "enable_ig_publish": ENABLE_IG_PUBLISH,
            "graph_version": f"v{GRAPH_VERSION}",
        },
        "result": {"posted_count": posted_count, "results": results}
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

if os.getenv("RUN_MODE", "A").upper() == "B":
    print(">>> RUN_MODE=B detectado. Ejecutando Modo B.")
    from ugc_mode_b import run_mode_b
    run_mode_b()
    raise SystemExit(0)

if __name__ == "__main__":
    accounts = load_accounts()
    if not accounts:
        raise RuntimeError("No se encontraron cuentas. Falta accounts.json o está vacío.")

    # sanity prints (sin revelar secretos)
    print("ENV CHECK:")
    print(" - R2_PUBLIC_BASE_URL:", (env_nonempty("R2_PUBLIC_BASE_URL", R2_PUBLIC_BASE_URL) or "")[:60])
    print(" - OPENAI_MODEL:", env_nonempty("OPENAI_MODEL", OPENAI_MODEL))
    print(" - ENABLE_IG_PUBLISH:", ENABLE_IG_PUBLISH)
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
