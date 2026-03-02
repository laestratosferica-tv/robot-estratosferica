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

# OpenAI SDK (para texto en Modo A)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

print("RUNNING MEDIA ENGINE (Threads REAL + IG REEL AUTO + IG PUBLISH + Multi-account via accounts.json)")

# =========================
# Helpers env
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

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_now() -> str:
    return now_utc().isoformat()

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
OPENAI_TTS_MODEL = env_nonempty("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL") or "").rstrip("/")

THREADS_GRAPH = env_nonempty("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")
THREADS_USER_ACCESS_TOKEN = env_nonempty("THREADS_USER_ACCESS_TOKEN")

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

POST_RETRY_MAX = env_int("POST_RETRY_MAX", 2)
POST_RETRY_SLEEP = env_float("POST_RETRY_SLEEP", 2.0)

CONTAINER_WAIT_TIMEOUT = env_int("CONTAINER_WAIT_TIMEOUT", 120)
CONTAINER_POLL_INTERVAL = env_float("CONTAINER_POLL_INTERVAL", 2.0)

# Instagram Graph API
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", False)
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
IG_USER_ID = env_nonempty("IG_USER_ID")
GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

# Reels generation
ENABLE_REELS = env_bool("ENABLE_REELS", True)
REEL_SECONDS = env_int("REEL_SECONDS", 15)
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)

DEFAULT_ASSET_BG = env_nonempty("ASSET_BG", "assets/bg.jpg")
DEFAULT_ASSET_LOGO = env_nonempty("ASSET_LOGO", "assets/logo.png")
DEFAULT_ASSET_MUSIC = env_nonempty("ASSET_MUSIC", "assets/music.mp3")  # opcional
FONT_BOLD = env_nonempty("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# =========================
# Runway (Modo A “animado”)
# =========================
RUNWAY_API_KEY = env_nonempty("RUNWAY_API_KEY")
RUNWAY_MODEL = env_nonempty("RUNWAY_MODEL", "gen4_turbo")  # ajustable
RUNWAY_SECONDS = env_int("RUNWAY_SECONDS", 5)
RUNWAY_RATIO = env_nonempty("RUNWAY_RATIO", "9:16")  # vertical

MODEA_STYLE_ENABLE = env_bool("MODEA_STYLE_ENABLE", True)
MODEA_LOGO_PCT = env_int("MODEA_LOGO_PCT", 40)
MODEA_NOLOGO_PCT = env_int("MODEA_NOLOGO_PCT", 60)
MODEA_VOICE_PCT = env_int("MODEA_VOICE_PCT", 25)
MODEA_MUSIC_PCT = env_int("MODEA_MUSIC_PCT", 70)

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

def upload_bytes_to_r2_public(file_bytes: bytes, ext: str, prefix: str, content_type: str) -> str:
    if not R2_PUBLIC_BASE_URL.startswith("http"):
        raise RuntimeError("R2_PUBLIC_BASE_URL inválido o vacío (debe empezar por https://)")
    s3 = r2_client()
    h = hashlib.sha1(file_bytes).hexdigest()[:16]
    prefix = (prefix or "").strip().strip("/")
    key = f"{prefix}/{h}{ext}" if prefix else f"{h}{ext}"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_bytes, ContentType=content_type)
    return f"{R2_PUBLIC_BASE_URL}/{key}"

def upload_image_bytes_to_r2_public(image_bytes: bytes, ext: str, prefix: str) -> str:
    content_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")
    return upload_bytes_to_r2_public(image_bytes, ext, prefix, content_type)

def upload_video_mp4_to_r2_public(video_bytes: bytes, prefix: str) -> str:
    return upload_bytes_to_r2_public(video_bytes, ".mp4", prefix, "video/mp4")

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
# RSS / images extraction (+ YouTube thumbnail fix)
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

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})")

def youtube_thumb_candidates(url: str) -> List[str]:
    m = YOUTUBE_ID_RE.search(url or "")
    if not m:
        return []
    vid = m.group(1)
    # orden de calidad
    return [
        f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
        f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg",
    ]

def extract_best_images(page_url: str, max_images: int = 5) -> List[str]:
    # ✅ Parche brutal para YouTube feeds (si el link es youtube)
    if "youtube.com" in (page_url or "") or "youtu.be" in (page_url or ""):
        return youtube_thumb_candidates(page_url)[:max_images]

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
# OpenAI (text) + TTS (HTTP)
# =========================
def openai_client():
    if not OpenAI:
        raise RuntimeError("No se pudo importar OpenAI. Revisa requirements.txt (openai).")
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_text(prompt: str) -> str:
    model = OPENAI_MODEL or "gpt-4.1-mini"
    client = openai_client()
    try:
        resp = client.responses.create(model=model, input=prompt)
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content or "").strip()

def openai_tts_mp3(text: str, voice: str = "alloy", instructions: str = "") -> bytes:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")
    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_TTS_MODEL,
        "voice": voice,
        "format": "mp3",
        "input": text[:400],
        "instructions": instructions[:400],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content

# =========================
# Copy prompts
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

RUNWAY_PROMPT = """Eres director creativo para un medio gamer LATAM.
Convierte esta noticia en un prompt corto y poderoso para animar una imagen estática (estilo reel).
Reglas:
- Estética gamer (neón, UI HUD, glitch, cyber, arcades, esports broadcast)
- Cámara: push-in suave + parallax + partículas + luces
- Sin sexual, sin desnudos, sin violencia extrema, sin gore, sin drogas.
- 1 sola escena, 5 segundos.
Salida: SOLO el prompt, sin comillas.

Noticia: {title}
"""

VOICE_PROMPT = """Eres caster de esports LATAM.
Haz una narración de 1 frase MUY corta, con hook fuerte, sobre esta noticia.
Sin inventar datos. Sin marcas raras.
Termina en pregunta.
Noticia: {title}
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
# Threads length guard
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
# Threads API
# =========================
def threads_create_container_image(user_id: str, access_token: str, text: str, image_url: str) -> str:
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

# =========================
# IG publish (Reels)
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

def ig_publish_reel(ig_user_id: str, access_token: str, video_url: str, caption: str) -> Dict[str, Any]:
    j = ig_api_post(
        f"{ig_user_id}/media",
        {"media_type": "REELS", "video_url": video_url, "caption": caption, "share_to_feed": "true", "access_token": access_token},
    )
    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG reels create failed: {j}")
    ig_wait_container(creation_id, access_token, timeout_sec=900)
    return ig_publish_media(ig_user_id, access_token, creation_id)

# =========================
# Reel building blocks
# =========================
def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"Falta {label} en repo: {path}")

def ffmpeg_loop_to_seconds(in_mp4: str, out_mp4: str, target_seconds: int) -> None:
    # loop video hasta target_seconds (sin audio)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-stream_loop", "-1",
        "-i", in_mp4,
        "-t", str(int(target_seconds)),
        "-vf", f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,crop={REEL_W}:{REEL_H},fps=30",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        out_mp4
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg loop failed:\n{(p.stderr or '')[:4000]}")

def ffmpeg_add_audio_mix(in_mp4: str, out_mp4: str, music_mp3: Optional[str], voice_mp3: Optional[str]) -> None:
    # si no hay audio, solo copy
    if not music_mp3 and not voice_mp3:
        cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-i", in_mp4, "-c:v", "copy", "-an", out_mp4]
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        return

    inputs = ["-i", in_mp4]
    idx = 1
    if music_mp3 and os.path.exists(music_mp3):
        inputs += ["-i", music_mp3]
        idx += 1
    if voice_mp3 and os.path.exists(voice_mp3):
        inputs += ["-i", voice_mp3]
        idx += 1

    # armamos filtros
    # 0:v es video
    # 0:a no existe, así que solo usamos externos y listo
    parts = []
    mix_inputs = []
    cur = 1
    if music_mp3 and os.path.exists(music_mp3):
        parts.append(f"[{cur}:a]volume=0.13[am]")
        mix_inputs.append("[am]")
        cur += 1
    if voice_mp3 and os.path.exists(voice_mp3):
        parts.append(f"[{cur}:a]volume=1.0[av]")
        mix_inputs.append("[av]")
        cur += 1

    filter_complex = ";".join(parts + ["".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:duration=shortest:dropout_transition=2[aout]"])

    cmd = [
        "ffmpeg","-y","-nostdin","-hide_banner","-loglevel","error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map","0:v",
        "-map","[aout]",
        "-c:v","copy",
        "-c:a","aac","-b:a","128k",
        "-shortest",
        out_mp4
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed:\n{(p.stderr or '')[:4000]}")

def generate_template_reel_mp4_bytes(headline: str, news_image_path: str, logo_path: str, bg_path: str, seconds: int, cta_text: Optional[str]) -> bytes:
    _require_file(bg_path, "ASSET_BG")
    _require_file(logo_path, "ASSET_LOGO")
    if not os.path.exists(news_image_path):
        raise RuntimeError(f"Falta news image local: {news_image_path}")
    if not os.path.exists(FONT_BOLD):
        raise RuntimeError(f"Falta FONT_BOLD en runner: {FONT_BOLD}")

    headline_clean = (headline or "").strip().replace("\n", " ")[:140]
    cta = (cta_text or "Sigue para más").strip()

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

        cmd += ["-filter_complex", vf, "-map", "[vout]", "-an"]
        cmd += ["-t", str(seconds), "-r", "30", "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", out_mp4]

        p = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg template falló:\n{(p.stderr or '')[:4000]}")

        with open(out_mp4, "rb") as f:
            return f.read()

# =========================
# Runway: Image -> Video (5s animado)
# =========================
def runway_image_to_video_mp4(image_url_https: str, prompt: str, seconds: int, ratio: str) -> bytes:
    """
    Implementación HTTP simple:
    - Crea task
    - Poll status
    - Baja output mp4

    Nota: Runway modera contenido y puede devolver FAILED con failureCode/failure si se modera. :contentReference[oaicite:2]{index=2}
    Nota: URLs deben ser HTTPS, con headers correctos y NO redirects. :contentReference[oaicite:3]{index=3}
    """
    if not RUNWAY_API_KEY:
        raise RuntimeError("RUNWAY_API_KEY no está configurado")
    # endpoints base (según docs oficiales)
    base = "https://api.dev.runwayml.com/v1"

    headers = {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "Content-Type": "application/json",
    }

    # Task create (image-to-video)
    # OJO: los campos exactos dependen del modelo; esto cubre el caso típico.
    create_payload = {
        "model": RUNWAY_MODEL,
        "input": {
            "prompt": (prompt or "")[:800],
            "image": image_url_https,
            "duration": int(seconds),
            "ratio": ratio,
        },
        # Por defecto Runway usa moderación "auto". :contentReference[oaicite:4]{index=4}
    }

    r = requests.post(f"{base}/tasks", headers=headers, json=create_payload, timeout=60)
    _raise_meta_error(r, "RUNWAY CREATE TASK")
    task = r.json()
    task_id = task.get("id") or task.get("taskId") or task.get("task_id")
    if not task_id:
        raise RuntimeError(f"Runway create no devolvió id: {task}")

    # Poll
    start = time.time()
    while time.time() - start < 600:
        rr = requests.get(f"{base}/tasks/{task_id}", headers=headers, timeout=30)
        _raise_meta_error(rr, "RUNWAY TASK STATUS")
        tj = rr.json()
        status = (tj.get("status") or "").upper()
        if status in ("SUCCEEDED", "SUCCESS", "COMPLETED"):
            out = tj.get("output") or {}
            # output puede traer urls
            # intentamos encontrar mp4 url
            mp4_url = None
            if isinstance(out, dict):
                for k in ["video", "video_url", "url", "mp4"]:
                    if out.get(k):
                        mp4_url = out.get(k)
                        break
                if not mp4_url and out.get("assets") and isinstance(out["assets"], list):
                    for a in out["assets"]:
                        u = a.get("url") if isinstance(a, dict) else None
                        if u and str(u).lower().endswith(".mp4"):
                            mp4_url = u
                            break
            if not mp4_url:
                # fallback: buscar cualquier string url mp4 en json
                s = json.dumps(tj)
                m = re.search(r"https?://[^\"']+\.mp4", s)
                mp4_url = m.group(0) if m else None

            if not mp4_url:
                raise RuntimeError(f"Runway task success pero no encontré mp4_url: {tj}")

            v = requests.get(mp4_url, timeout=120)
            v.raise_for_status()
            return v.content

        if status in ("FAILED", "ERROR"):
            # Runway si modera: status FAILED y failure/failureCode. :contentReference[oaicite:5]{index=5}
            raise RuntimeError(f"Runway task FAILED: {tj}")

        time.sleep(2.5)

    raise TimeoutError("Runway task timeout (10 min)")

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
# State helpers
# =========================
def parse_iso(dt: str) -> datetime:
    return datetime.fromisoformat(dt.replace("Z", "+00:00"))

def days_since(dt_iso: str) -> int:
    try:
        dt = parse_iso(dt_iso)
        return int((now_utc() - dt).total_seconds() // 86400)
    except Exception:
        return 999999

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
# Ruleta visual Modo A
# =========================
def roulette_bool(pct_true: int) -> bool:
    pct_true = max(0, min(100, int(pct_true)))
    return random.randint(1, 100) <= pct_true

def pick_voice_settings() -> Tuple[str, str]:
    presets = [
        ("nova", "Voz femenina LATAM, caster esports, rápida, hype."),
        ("onyx", "Voz masculina LATAM, narrador épico tipo tráiler, intensa."),
        ("shimmer", "Voz femenina humor gamer, sarcástica amable."),
        ("alloy", "Voz neutra, análisis competitivo, segura."),
    ]
    return random.choice(presets)

# =========================
# Main per account
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
    threads_media_prefix = (r2_cfg.get("threads_media_prefix", f"threads_media/{account_id}") or "").strip().strip("/")
    reels_prefix = (r2_cfg.get("reels_prefix", f"ugc/reels/{account_id}") or "").strip().strip("/")

    if auto_post_limit <= 0 or not rss_feeds:
        print(f"Cuenta {account_id} está apagada (auto_post_limit=0 o rss_feeds vacío). Saltando ✅")
        return {"generated_at": iso_now(), "account_id": account_id, "skipped": True, "reason": "disabled_or_no_feeds"}

    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles(rss_feeds, max_per_feed=max_per_feed, shuffle=shuffle)
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={max_per_feed}, SHUFFLE={shuffle})")

    processed = articles[:max_ai_items]
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

        # Texto Threads + caption IG
        text = build_threads_text(item, mode=mode)
        caption = build_instagram_caption(item, link)

        # Buscar imagen (con fix YouTube thumbs)
        img_candidates = extract_best_images(link, max_images=5)
        if not img_candidates:
            print("No se encontró imagen. Se omite.")
            processed = [x for x in processed if x.get("link") != link]
            continue

        chosen_img = None
        chosen_bytes = None
        chosen_ext = None
        for u in img_candidates:
            try:
                b, ext = download_image_bytes(u)
                chosen_img = u
                chosen_bytes = b
                chosen_ext = ext
                break
            except Exception:
                continue

        if not chosen_img or not chosen_bytes or not chosen_ext:
            print("Todas las imágenes del artículo fallaron. Se omite.")
            processed = [x for x in processed if x.get("link") != link]
            continue

        # Rehost imagen en R2 (para Threads y para Runway)
        img_r2 = upload_image_bytes_to_r2_public(chosen_bytes, chosen_ext, prefix=threads_media_prefix)
        print("Imagen rehost R2:", img_r2)

        # Threads publish
        threads_res = {"ok": True, "dry_run": True}
        if not dry_run:
            if not THREADS_USER_ACCESS_TOKEN:
                raise RuntimeError("Falta THREADS_USER_ACCESS_TOKEN")
            container_id = threads_create_container_image(threads_user_id, THREADS_USER_ACCESS_TOKEN, text, img_r2)
            print("Container created:", container_id)
            threads_wait_container(container_id, THREADS_USER_ACCESS_TOKEN)
            pub = threads_publish(threads_user_id, THREADS_USER_ACCESS_TOKEN, container_id)
            print("Threads publish response:", pub)
            threads_res = {"ok": True, "container": {"id": container_id}, "publish": pub, "image_url": img_r2}

        # Reels (Modo A) — ahora animado con Runway (si hay key)
        reel_url = None
        if ENABLE_REELS:
            try:
                use_logo = roulette_bool(MODEA_LOGO_PCT) and not roulette_bool(MODEA_NOLOGO_PCT)  # balance simple
                use_music = roulette_bool(MODEA_MUSIC_PCT) and os.path.exists(asset_music)
                use_voice = roulette_bool(MODEA_VOICE_PCT)

                voice_mp3_path = None
                music_mp3_path = asset_music if use_music else None

                # Voz IA (opcional)
                narration = ""
                if use_voice:
                    narration = openai_text(VOICE_PROMPT.format(title=item.get("title",""))).strip()[:220]
                    vname, vinstr = pick_voice_settings()
                    try:
                        vbytes = openai_tts_mp3(narration, voice=vname, instructions=vinstr)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                            tf.write(vbytes)
                            voice_mp3_path = tf.name
                    except Exception as e:
                        print("AVISO: TTS falló, sigo sin voz:", str(e))
                        voice_mp3_path = None

                with tempfile.TemporaryDirectory() as td:
                    # 1) Crear clip base 5s (Runway) o fallback template
                    base_mp4 = os.path.join(td, "base.mp4")

                    if RUNWAY_API_KEY and MODEA_STYLE_ENABLE:
                        runway_prompt = openai_text(RUNWAY_PROMPT.format(title=item.get("title",""))).strip()
                        print("Runway prompt:", runway_prompt[:140])
                        mp4_bytes = runway_image_to_video_mp4(img_r2, runway_prompt, seconds=RUNWAY_SECONDS, ratio=RUNWAY_RATIO)
                        with open(base_mp4, "wb") as f:
                            f.write(mp4_bytes)
                    else:
                        # fallback template
                        news_img_path = os.path.join(td, "news.jpg")
                        with open(news_img_path, "wb") as f:
                            f.write(chosen_bytes)
                        mp4_bytes = generate_template_reel_mp4_bytes(item.get("title",""), news_img_path, asset_logo, asset_bg, RUNWAY_SECONDS, cta_text)
                        with open(base_mp4, "wb") as f:
                            f.write(mp4_bytes)

                    # 2) Loop hasta 15s
                    loop_mp4 = os.path.join(td, "loop.mp4")
                    ffmpeg_loop_to_seconds(base_mp4, loop_mp4, target_seconds=REEL_SECONDS)

                    # 3) Agregar audio (música + voz) si aplica
                    final_mp4 = os.path.join(td, "final.mp4")
                    ffmpeg_add_audio_mix(loop_mp4, final_mp4, music_mp3_path, voice_mp3_path)

                    # 4) Subir a R2
                    vb = open(final_mp4, "rb").read()
                    reel_url = upload_video_mp4_to_r2_public(vb, prefix=reels_prefix)
                    print("Reel URL R2:", reel_url)

                    # 5) Publicar IG
                    ig_res = {"ok": True, "dry_run": True}
                    if ENABLE_IG_PUBLISH and IG_USER_ID and IG_ACCESS_TOKEN and not dry_run:
                        ig_pub = ig_publish_reel(IG_USER_ID, IG_ACCESS_TOKEN, reel_url, caption)
                        ig_res = ig_pub
                    else:
                        print("[DRY_RUN] IG disabled or dry_run, no publico.")

                # cleanup temp voice
                try:
                    if voice_mp3_path and os.path.exists(voice_mp3_path):
                        os.unlink(voice_mp3_path)
                except Exception:
                    pass

            except Exception as e:
                print("AVISO: Reel generation/publish falló (no rompe):", str(e))

        # Guardar state
        if not dry_run:
            mark_posted(state, link)
            save_threads_state(state_key, state)
            posted_count += 1

        results.append({
            "link": link,
            "mode": mode,
            "threads": threads_res,
            "ig": ({"video_url": reel_url} if reel_url else {}),
            "ig_kind": "reel",
            "dry_run": dry_run,
        })

        break

    run_payload = {
        "generated_at": iso_now(),
        "account_id": account_id,
        "settings": {
            "enable_reels": ENABLE_REELS,
            "reel_seconds": REEL_SECONDS,
            "enable_ig_publish": ENABLE_IG_PUBLISH,
            "graph_version": f"v{GRAPH_VERSION}",
            "runway_enabled": bool(RUNWAY_API_KEY),
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
if __name__ == "__main__":
    accounts = load_accounts()
    if not accounts:
        raise RuntimeError("No se encontraron cuentas. Falta accounts.json o está vacío.")

    print("ENV CHECK:")
    print(" - RUN_MODE:", RUN_MODE)
    print(" - ENABLE_IG_PUBLISH:", ENABLE_IG_PUBLISH)
    print(" - ENABLE_REELS:", ENABLE_REELS)
    print(" - RUNWAY enabled:", bool(RUNWAY_API_KEY))
    print(" - R2_PUBLIC_BASE_URL set:", bool(R2_PUBLIC_BASE_URL))
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
