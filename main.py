import os
import re
import json
import time
import hashlib
import random
import subprocess
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


print("RUNNING MEDIA ENGINE (Threads REAL + IG Queue + REEL AUTO + Multi-account via accounts.json)")

# =========================
# GLOBAL ENV
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# TTS
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

R2_PUBLIC_BASE_URL = os.getenv(
    "R2_PUBLIC_BASE_URL",
    "https://pub-8937244ee725495691514507bb8f431e.r2.dev"
).rstrip("/")

THREADS_GRAPH = os.getenv("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")
THREADS_USER_ACCESS_TOKEN = os.getenv("THREADS_USER_ACCESS_TOKEN")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
POST_RETRY_MAX = int(os.getenv("POST_RETRY_MAX", "2"))
POST_RETRY_SLEEP = float(os.getenv("POST_RETRY_SLEEP", "2"))

CONTAINER_WAIT_TIMEOUT = int(os.getenv("CONTAINER_WAIT_TIMEOUT", "120"))
CONTAINER_POLL_INTERVAL = float(os.getenv("CONTAINER_POLL_INTERVAL", "2"))

IG_CAROUSEL_MAX_IMAGES = int(os.getenv("IG_CAROUSEL_MAX_IMAGES", "5"))

# MODO A: publicar sí o sí (sin verificación estricta / sin trends)
VERIFY_NEWS = os.getenv("VERIFY_NEWS", "false").lower() == "true"
ENABLE_TRENDS = os.getenv("ENABLE_TRENDS", "false").lower() == "true"

# Reel auto
ENABLE_REELS = os.getenv("ENABLE_REELS", "true").lower() == "true"
REEL_SECONDS = int(os.getenv("REEL_SECONDS", "15"))
REEL_W = int(os.getenv("REEL_W", "1080"))
REEL_H = int(os.getenv("REEL_H", "1920"))
ASSET_BG = os.getenv("ASSET_BG", "assets/bg.jpg")
ASSET_LOGO = os.getenv("ASSET_LOGO", "assets/logo.png")
ASSET_MUSIC = os.getenv("ASSET_MUSIC", "assets/music.mp3")  # opcional

# Font for drawtext (GitHub ubuntu runners usually have DejaVu)
FONT_BOLD = os.getenv("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

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

def _threads_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def _raise_meta_error(r: requests.Response, label: str = "HTTP") -> None:
    if r.status_code < 400:
        return

    print(f"\n====== {label} ERROR ======")
    print("URL:", r.request.url)
    print("METHOD:", r.request.method)
    print("STATUS:", r.status_code)
    if r.request.body:
        body = r.request.body
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        print("REQUEST BODY:", body)
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
    if "mpeg" in ct or "mp3" in ct:
        return ".mp3"
    return ".bin"

def upload_bytes_to_r2_public(data_bytes: bytes, ext: str, prefix: str, content_type: Optional[str] = None) -> str:
    s3 = r2_client()
    h = hashlib.sha1(data_bytes).hexdigest()[:16]
    key = f"{prefix}/{h}{ext}"

    if not content_type:
        content_type = {
            ".png": "image/png",
            ".webp": "image/webp",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".mp4": "video/mp4",
            ".mp3": "audio/mpeg",
        }.get(ext, "application/octet-stream")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data_bytes,
        ContentType=content_type,
    )

    url = f"{R2_PUBLIC_BASE_URL}/{key}"

    head = requests.head(url, timeout=10, allow_redirects=True)
    if head.status_code != 200:
        raise RuntimeError(f"R2 public URL no accesible (status {head.status_code}): {url}")

    ct = (head.headers.get("Content-Type") or "").lower()
    # Accept image/video/audio depending on type
    if ext in (".png", ".jpg", ".jpeg", ".webp") and "image" not in ct:
        raise RuntimeError(f"R2 URL no parece imagen (Content-Type={ct}): {url}")
    if ext == ".mp4" and "video" not in ct:
        # some CDNs may return octet-stream; allow it
        if "octet-stream" not in ct:
            raise RuntimeError(f"R2 URL no parece video (Content-Type={ct}): {url}")

    return url

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
                    if img.startswith("/"):
                        img = urljoin(page_url, img)
                    if img not in found:
                        found.append(img)

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

def download_bytes(url: str) -> Tuple[bytes, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    ext = _guess_ext_from_content_type(r.headers.get("Content-Type", ""))
    return r.content, ext

def download_image_bytes(image_url: str) -> Tuple[bytes, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": image_url,
    }
    r = requests.get(image_url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    ext = _guess_ext_from_content_type(r.headers.get("Content-Type", ""))
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
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
                    raw.append({
                        "title": title.strip(),
                        "link": link.strip(),
                        "published": published,
                        "feed": feed,
                    })
        except Exception:
            continue

    # de-dup por link
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for a in raw:
        if a["link"] not in seen:
            seen.add(a["link"])
            deduped.append(a)

    # balance por feed
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
# Simple content filters (avoid guides/codes)
# =========================

BAD_TITLE_PATTERNS = [
    r"\bhow to\b",
    r"\bguide\b",
    r"\bcodes?\b",
    r"\bquest\b",
    r"\bwalkthrough\b",
    r"\btips?\b",
    r"\bbest\b.*\bsettings\b",
]

def is_bad_title(title: str) -> bool:
    t = (title or "").lower()
    return any(re.search(p, t) for p in BAD_TITLE_PATTERNS)

# =========================
# OpenAI (text + TTS)
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

def openai_tts_to_mp3(text: str) -> bytes:
    """
    Returns MP3 bytes from OpenAI TTS.
    """
    client = openai_client()
    # Newer SDK supports audio.speech.create
    try:
        audio = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            format="mp3",
        )
        # audio might be bytes-like or have .read()
        if hasattr(audio, "read"):
            return audio.read()
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        # some SDK returns an object with .content
        if hasattr(audio, "content"):
            return audio.content
    except Exception as e:
        raise RuntimeError(f"TTS falló: {e}")
    raise RuntimeError("TTS falló: respuesta inesperada")

def build_threads_text(item: Dict[str, Any], mode: str = "new") -> str:
    title = item.get("title", "")
    link = item.get("link", "")

    if mode == "repost":
        prompt = f"""
Eres editor para una cuenta de Threads sobre esports/gaming (español LATAM).
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
Eres editor para una cuenta de Threads sobre esports/gaming (español LATAM).
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
    prompt = f"""
Eres editor de Instagram (esports/gaming) para público en ESPAÑOL LATAM.
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

def build_reel_script(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns: {"hook":..., "body":..., "cta":..., "voice":...}
    """
    title = item.get("title", "")
    link = item.get("link", "")
    prompt = f"""
Eres guionista de reels gaming para LATAM.
Con base en el título, crea un guion ultra corto para 15s:
- hook (máx 8 palabras)
- body (1 frase clara)
- cta (máx 7 palabras)
- voiceover: 2-3 frases naturales (máx 40 palabras total), tono hype gamer, sin decir "según", sin sonar noticiero formal.
No inventes datos. Si no hay datos del link, habla en general del anuncio/noticia por el título.
Entrega JSON con keys: hook, body, cta, voice.
Título: {title}
Link: {link}
"""
    raw = openai_text(prompt)
    # try parse json
    try:
        j = json.loads(raw)
        return {
            "hook": str(j.get("hook", "")).strip(),
            "body": str(j.get("body", "")).strip(),
            "cta": str(j.get("cta", "")).strip(),
            "voice": str(j.get("voice", "")).strip(),
        }
    except Exception:
        # fallback: simple
        return {
            "hook": "HOY EN GAMING",
            "body": title[:90],
            "cta": "¿Qué opinas?",
            "voice": f"Atención gamers. {title}. ¿Qué opinas de esto?",
        }

# =========================
# Threads API (real)
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

def threads_publish_text_image(user_id: str, access_token: str, dry_run: bool, text: str, image_url_from_news: str, threads_media_prefix: str) -> Dict[str, Any]:
    if dry_run:
        print("[DRY_RUN] Threads post:", text)
        print("[DRY_RUN] Image source:", image_url_from_news)
        return {"ok": True, "dry_run": True}

    if not access_token:
        raise RuntimeError("Falta THREADS_USER_ACCESS_TOKEN")

    img_bytes, ext = download_image_bytes(image_url_from_news)
    r2_url = upload_bytes_to_r2_public(img_bytes, ext, prefix=threads_media_prefix, content_type=None)
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
# Reel generator (FFmpeg)
# =========================

def _ffmpeg_exists() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        return r.returncode == 0
    except Exception:
        return False

def _safe_drawtext(text: str) -> str:
    # escape for ffmpeg drawtext
    t = (text or "").replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def create_reel_mp4(
    *,
    account_id: str,
    title: str,
    article_image_url: str,
    reel_script: Dict[str, str],
    tmp_dir: str = "/tmp"
) -> str:
    """
    Creates mp4, returns local path.
    """
    if not _ffmpeg_exists():
        raise RuntimeError("ffmpeg no está disponible en el runner.")

    if not os.path.exists(ASSET_BG):
        raise RuntimeError(f"Falta fondo: {ASSET_BG}. Súbelo al repo.")
    if not os.path.exists(ASSET_LOGO):
        raise RuntimeError(f"Falta logo: {ASSET_LOGO}. Súbelo al repo.")

    # download article image
    img_bytes, img_ext = download_image_bytes(article_image_url)
    article_img_path = os.path.join(tmp_dir, f"article_{account_id}{img_ext}")
    with open(article_img_path, "wb") as f:
        f.write(img_bytes)

    # TTS
    voice_text = reel_script.get("voice") or f"Atención gamers. {title}. ¿Qué opinas?"
    tts_mp3_bytes = openai_tts_to_mp3(voice_text)
    voice_path = os.path.join(tmp_dir, f"voice_{account_id}.mp3")
    with open(voice_path, "wb") as f:
        f.write(tts_mp3_bytes)

    # Optional music
    music_exists = os.path.exists(ASSET_MUSIC)

    # Output
    out_path = os.path.join(tmp_dir, f"reel_{account_id}_{now_utc().strftime('%Y%m%d_%H%M%S')}.mp4")

    hook = _safe_drawtext(reel_script.get("hook") or "HOY EN GAMING")
    body = _safe_drawtext(reel_script.get("body") or title)
    cta = _safe_drawtext(reel_script.get("cta") or "¿Qué opinas?")

    # Timing
    # 0-4 hook, 4-11 body, 11-15 cta
    # Build filter_complex
    # Base bg image loops, then overlays logo and article card and text.
    # Article card: scaled to fit center.
    logo_scale_w = 500  # adjust
    card_w = 900
    card_h = 520

    # drawtext boxes for readability
    # We'll use x centered and y positions
    fc = []
    fc.append(f"[0:v]scale={REEL_W}:{REEL_H},format=yuv420p[bg]")
    fc.append(f"[2:v]scale={logo_scale_w}:-1[logo]")
    fc.append(f"[1:v]scale={card_w}:{card_h}:force_original_aspect_ratio=decrease,pad={card_w}:{card_h}:(ow-iw)/2:(oh-ih)/2:color=black@0,format=rgba[card]")

    # Overlay card then logo
    fc.append(f"[bg][card]overlay=(W-w)/2:(H-h)/2-140:enable='between(t,0,{REEL_SECONDS})'[v1]")
    fc.append(f"[v1][logo]overlay=(W-w)/2:120:enable='between(t,0,{REEL_SECONDS})'[v2]")

    # Text layers
    # Hook
    fc.append(
        f"[v2]drawtext=fontfile={FONT_BOLD}:text='{hook}':fontsize=72:fontcolor=white:"
        f"x=(w-text_w)/2:y=740:box=1:boxcolor=black@0.35:boxborderw=18:"
        f"enable='between(t,0,4)'[v3]"
    )
    # Body
    fc.append(
        f"[v3]drawtext=fontfile={FONT_BOLD}:text='{body}':fontsize=44:fontcolor=white:"
        f"x=(w-text_w)/2:y=860:box=1:boxcolor=black@0.35:boxborderw=18:"
        f"enable='between(t,4,11)'[v4]"
    )
    # CTA
    fc.append(
        f"[v4]drawtext=fontfile={FONT_BOLD}:text='{cta}':fontsize=56:fontcolor=white:"
        f"x=(w-text_w)/2:y=1560:box=1:boxcolor=black@0.35:boxborderw=18:"
        f"enable='between(t,11,{REEL_SECONDS})'[vout]"
    )

    filter_complex = ";".join(fc)

    # Audio mixing:
    # input 3 = voice mp3
    # input 4 = music mp3 (optional)
    # We'll normalize voice and (optional) mix music low volume.
    if music_exists:
        audio_fc = (
            "[3:a]aformat=fltp:44100:stereo,volume=1.0[voice];"
            "[4:a]aformat=fltp:44100:stereo,volume=0.18[music];"
            "[voice][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        full_fc = filter_complex + ";" + audio_fc
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(REEL_SECONDS), "-i", ASSET_BG,          # 0
            "-i", article_img_path,                                         # 1
            "-i", ASSET_LOGO,                                                # 2
            "-i", voice_path,                                                # 3
            "-stream_loop", "-1", "-i", ASSET_MUSIC,                         # 4
            "-filter_complex", full_fc,
            "-map", "[vout]",
            "-map", "[aout]",
            "-t", str(REEL_SECONDS),
            "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            out_path
        ]
    else:
        audio_fc = "[3:a]aformat=fltp:44100:stereo,volume=1.0[aout]"
        full_fc = filter_complex + ";" + audio_fc
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(REEL_SECONDS), "-i", ASSET_BG,          # 0
            "-i", article_img_path,                                         # 1
            "-i", ASSET_LOGO,                                                # 2
            "-i", voice_path,                                                # 3
            "-filter_complex", full_fc,
            "-map", "[vout]",
            "-map", "[aout]",
            "-t", str(REEL_SECONDS),
            "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            out_path
        ]

    print("FFmpeg cmd:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFmpeg STDOUT:", r.stdout[-2000:])
        print("FFmpeg STDERR:", r.stderr[-4000:])
        raise RuntimeError("FFmpeg falló generando el reel.")

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 50_000:
        raise RuntimeError("Reel mp4 generado inválido.")

    return out_path

def upload_mp4_to_r2_public(local_path: str, prefix: str) -> str:
    with open(local_path, "rb") as f:
        data = f.read()
    return upload_bytes_to_r2_public(data, ".mp4", prefix=prefix, content_type="video/mp4")

# =========================
# Accounts loading
# =========================

def load_accounts() -> List[Dict[str, Any]]:
    """
    Si existe 'accounts.json' en el repo, lo usa.
    Si no existe, corre 1 cuenta default (estratosferica).
    """
    try:
        with open("accounts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "accounts" in data and isinstance(data["accounts"], list):
            return data["accounts"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Default 1 cuenta (fuentes mejores para “noticias”)
    return [{
        "account_id": "estratosferica",
        "rss_feeds": [
            # Gaming / esports
            "https://www.dexerto.com/feed/",
            "https://www.gamespot.com/feeds/news/",
            "https://www.pcgamer.com/rss/",
            "https://dotesports.com/feed",
            "https://www.videogameschronicle.com/feed/",
            "https://www.eurogamer.net/feed",
            "https://www.polygon.com/rss/index.xml",
            # Tech/hardware (para gamers)
            "https://www.theverge.com/rss/index.xml",
            "https://www.tomshardware.com/feeds/all",
        ],
        "max_per_feed": int(os.getenv("MAX_PER_FEED", "3")),
        "shuffle_articles": os.getenv("SHUFFLE_ARTICLES", "true").lower() == "true",
        "max_ai_items": int(os.getenv("MAX_AI_ITEMS", "20")),
        "threads": {
            "user_id": os.getenv("THREADS_USER_ID", "me"),
            "state_key": "accounts/estratosferica/threads_state.json",
            "auto_post": os.getenv("THREADS_AUTO_POST", "true").lower() == "true",
            "auto_post_limit": int(os.getenv("THREADS_AUTO_POST_LIMIT", "1")),
            "dry_run": os.getenv("THREADS_DRY_RUN", "false").lower() == "true",
            "repost_enable": os.getenv("REPOST_ENABLE", "true").lower() == "true",
            "repost_max_times": int(os.getenv("REPOST_MAX_TIMES", "3")),
            "repost_window_days": int(os.getenv("REPOST_WINDOW_DAYS", "7"))
        },
        "r2": {
            "threads_media_prefix": "threads_media/estratosferica",
            "ig_queue_prefix": "ugc/ig_queue/estratosferica",
            "reels_prefix": "ugc/reels/estratosferica"
        }
    }]

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

    # RSS
    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles(rss_feeds, max_per_feed=max_per_feed, shuffle=shuffle)
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={max_per_feed}, SHUFFLE={shuffle})")
    feeds_in_run = sorted(set([a.get("feed", "") for a in articles]))
    print("FEEDS EN ESTA CORRIDA:", feeds_in_run)
    print("TOTAL FEEDS:", len(feeds_in_run))

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

        # FILTRO: evita guías/códigos (más viral = noticias)
        if is_bad_title(item.get("title", "")):
            print("Saltando guía/códigos:", item.get("title"))
            processed = [x for x in processed if x.get("link") != link]
            continue

        # (Modo A) No forzamos verificación para que siempre publique
        if VERIFY_NEWS:
            pass

        # Imagen para Threads (al menos 1)
        imgs_1 = extract_best_images(link, max_images=2)
        if not imgs_1:
            print("No se encontró imagen (og/twitter/img). Se omite para evitar post sin imagen.")
            processed = [x for x in processed if x.get("link") != link]
            continue

        # Threads text
        text = build_threads_text(item, mode=mode)

        if not auto_post:
            print("THREADS_AUTO_POST desactivado. (No se publica)")
            results.append({"link": link, "mode": mode, "posted": False, "reason": "auto_post_off"})
            break

        print(f"Publicando en Threads ({label})...")
        try:
            threads_res = threads_publish_text_image(
                user_id=threads_user_id,
                access_token=THREADS_USER_ACCESS_TOKEN,
                dry_run=dry_run,
                text=text,
                image_url_from_news=imgs_1[0],
                threads_media_prefix=threads_media_prefix,
            )

            mark_posted(state, link)
            save_threads_state(state_key, state)

            posted_count += 1
            results.append({"link": link, "mode": mode, "posted": True, "threads": threads_res})
            print("Auto-post Threads: OK ✅")

            # =========================
            # REEL AUTO (mp4 + voice)
            # =========================
            reel_video_url = None
            reel_local_path = None
            reel_script = None
            try:
                if ENABLE_REELS:
                    print("Generando REEL automático (15s)...")
                    reel_script = build_reel_script(item)

                    # Use second image if exists, else first
                    article_img_for_reel = imgs_1[1] if len(imgs_1) > 1 else imgs_1[0]

                    reel_local_path = create_reel_mp4(
                        account_id=account_id,
                        title=item.get("title", ""),
                        article_image_url=article_img_for_reel,
                        reel_script=reel_script,
                    )
                    reel_video_url = upload_mp4_to_r2_public(reel_local_path, prefix=reels_prefix)
                    print("REEL subido a R2:", reel_video_url)
            except Exception as e:
                print("REEL: falló (no rompe el run):", str(e))

            # =========================
            # IG QUEUE
            # =========================
            try:
                candidates = extract_best_images(link, max_images=IG_CAROUSEL_MAX_IMAGES)

                r2_images: List[str] = []
                for img_url in candidates:
                    try:
                        b, ext = download_image_bytes(img_url)
                        r2_img = upload_bytes_to_r2_public(b, ext, prefix=threads_media_prefix, content_type=None)
                        if r2_img not in r2_images:
                            r2_images.append(r2_img)
                    except Exception:
                        continue
                    if len(r2_images) >= IG_CAROUSEL_MAX_IMAGES:
                        break

                threads_img = threads_res.get("image_url") if isinstance(threads_res, dict) else None
                if not r2_images and threads_img:
                    r2_images = [threads_img]

                has_video = bool(reel_video_url)
                ig_format = choose_ig_format(has_video=has_video, image_count=len(r2_images))
                ig_caption = build_instagram_caption(item, mode=mode, link=link)

                ig_payload = {
                    "created_at": iso_now(),
                    "account_id": account_id,
                    "source": {
                        "link": link,
                        "feed": item.get("feed"),
                        "mode": mode,
                        "title": item.get("title"),
                    },
                    "format": ig_format,  # reel | carousel | image
                    "caption": ig_caption,
                    "assets": {
                        "images": r2_images[:IG_CAROUSEL_MAX_IMAGES],
                        "reel_video_url": reel_video_url,
                        "reel_script": reel_script,  # útil para subtítulos/copy
                    },
                    "threads": {
                        "publish_id": (threads_res.get("publish") or {}).get("id") if isinstance(threads_res, dict) else None,
                        "image_url": threads_img,
                    },
                }

                ig_key = save_ig_queue_item(ig_queue_prefix, ig_payload)
                print("IG queue guardado en R2:", ig_key)
            except Exception as e:
                print("IG queue: falló (no rompe el run):", str(e))

        except Exception as e:
            print("Auto-post Threads: FALLÓ ❌")
            print("ERROR:", str(e))
            results.append({"link": link, "mode": mode, "posted": False, "error": str(e)})
            break

        processed = [x for x in processed if x.get("link") != link]

    run_payload = {
        "generated_at": iso_now(),
        "account_id": account_id,
        "mix": {
            "shuffle": shuffle,
            "max_per_feed": max_per_feed,
            "max_ai_items": max_ai_items,
        },
        "settings": {
            "verify_news": VERIFY_NEWS,
            "enable_trends": ENABLE_TRENDS,
            "enable_reels": ENABLE_REELS,
            "reel_seconds": REEL_SECONDS,
        },
        "result": {
            "posted_count": posted_count,
            "results": results,
        }
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

    all_results = []
    for cfg in accounts:
        account_id = cfg.get("account_id", "unknown")
        payload = run_account(cfg)
        run_key = save_run_payload(account_id, payload)
        all_results.append({"account_id": account_id, "run_key": run_key, "payload": payload})
        print("RUN COMPLETED:", account_id)

    print("\n===== SUMMARY =====")
    print(json.dumps({"runs": all_results}, ensure_ascii=False, indent=2))
