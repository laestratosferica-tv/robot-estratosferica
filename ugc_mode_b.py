# ugc_mode_b.py
import os
import re
import json
import time
import random
import hashlib
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import boto3
import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# ENV helpers
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
# Core settings
# =========================
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
REEL_FPS = env_int("REEL_FPS", 30)
REEL_SECONDS = env_int("REEL_SECONDS", 12)  # en B suele ir mejor 10-15

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

# R2
AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or "").rstrip("/")

INBOX_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")
OUT_PREFIX = (env_nonempty("UGC_OUT_PREFIX", "ugc/outputs/reels") or "ugc/outputs/reels").strip().strip("/")

# Visual style
ENABLE_HUD_OVERLAYS = env_bool("ENABLE_HUD_OVERLAYS", True)
HUD_DIR = env_nonempty("HUD_DIR", "assets/hud") or "assets/hud"
GLITCH_PROB = env_float("GLITCH_PROB", 0.45)   # prob de flashes glitch
SHAKE_PROB = env_float("SHAKE_PROB", 0.35)
GRAIN = env_float("GRAIN", 0.08)               # 0..0.2 recomendado
SAT = env_float("SAT", 1.20)
CONTRAST = env_float("CONTRAST", 1.15)
BRIGHT = env_float("BRIGHT", 0.02)

# Audio
ENABLE_MUSIC = env_bool("ENABLE_MUSIC", True)
MUSIC_DIR = env_nonempty("MUSIC_SEARCH_DIR", "assets") or "assets"
MUSIC_VOLUME = env_float("MUSIC_VOLUME", 0.20)
ORIG_VOLUME = env_float("ORIG_VOLUME", 1.00)

# Publish IG
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", False)
IG_USER_ID = env_nonempty("IG_USER_ID")
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

# OpenAI captions
OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")

# Runway (intro sorpresa)
RUNWAY_ENABLED = env_bool("RUNWAY_ENABLED", False)
RUNWAY_API_KEY = env_nonempty("RUNWAY_API_KEY")
RUNWAY_VERSION = env_nonempty("RUNWAY_VERSION", "2024-11-06")
RUNWAY_BASE = (env_nonempty("RUNWAY_BASE", "https://api.dev.runwayml.com") or "").rstrip("/")
RUNWAY_I2V_MODEL = env_nonempty("RUNWAY_I2V_MODEL", "gen4.5")
RUNWAY_INTRO_SECONDS = env_int("RUNWAY_INTRO_SECONDS", 3)
RUNWAY_INTRO_PROB = env_float("RUNWAY_INTRO_PROB", 0.55)
RUNWAY_TIMEOUT = env_int("RUNWAY_TIMEOUT", 420)
RUNWAY_POLL_SEC = env_int("RUNWAY_POLL_SEC", 6)

MP4_URL_RE = re.compile(r"https?://[^\s\"']+\.mp4[^\s\"']*", re.IGNORECASE)


# =========================
# R2 helpers
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

def list_r2_keys(prefix: str) -> List[str]:
    s3 = r2_client()
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": BUCKET_NAME, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        res = s3.list_objects_v2(**kwargs)
        for it in (res.get("Contents") or []):
            k = it.get("Key")
            if k and k.lower().endswith(".mp4"):
                keys.append(k)
        if res.get("IsTruncated"):
            token = res.get("NextContinuationToken")
        else:
            break
    keys.sort()
    return keys

def get_r2_bytes(key: str) -> bytes:
    s3 = r2_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return obj["Body"].read()

def put_r2_bytes(key: str, data: bytes, content_type: str = "video/mp4") -> str:
    s3 = r2_client()
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    if R2_PUBLIC_BASE_URL.startswith("http"):
        return f"{R2_PUBLIC_BASE_URL}/{key}"
    return key

def delete_r2_key(key: str) -> None:
    s3 = r2_client()
    s3.delete_object(Bucket=BUCKET_NAME, Key=key)


# =========================
# Local helpers
# =========================
def ffprobe_has_audio(path: str) -> bool:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "json", path],
            capture_output=True, text=True, timeout=20, check=False
        )
        if p.returncode != 0:
            return False
        j = json.loads(p.stdout or "{}")
        return bool((j.get("streams") or []))
    except Exception:
        return False

def pick_music_file() -> Optional[str]:
    if not ENABLE_MUSIC:
        return None
    if not os.path.isdir(MUSIC_DIR):
        return None
    cands = []
    for root, _, files in os.walk(MUSIC_DIR):
        for f in files:
            if f.lower().endswith(".mp3"):
                p = os.path.join(root, f)
                try:
                    if os.path.getsize(p) > 50_000:
                        cands.append(p)
                except Exception:
                    pass
    if not cands:
        return None
    return random.choice(cands)

def list_hud_overlays() -> List[str]:
    if not (ENABLE_HUD_OVERLAYS and os.path.isdir(HUD_DIR)):
        return []
    out = []
    for f in os.listdir(HUD_DIR):
        if f.lower().endswith(".png"):
            out.append(os.path.join(HUD_DIR, f))
    out.sort()
    return out

def safe_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:10]


# =========================
# Caption (polémico sano)
# =========================
CAPTION_TEMPLATES = [
    "¿Esto es skill o pura suerte? 😭🔥\nTeam {a} vs Team {b}… ¿de qué lado estás? 👇",
    "Esto debería ser ilegal en ranked 💀🎮\n¿Lo aplaudes o lo reportas? 👇",
    "El gaming en 2026: {a} o {b} 😤\n¿Quién tiene la razón? 👇",
    "Nadie habla de esto… y es el verdadero problema 😶‍🌫️\n¿Exagero o es real? 👇",
    "Si te pasa esto, ¿sigues jugando o cierras el juego? 😭\nCuéntame tu peor tilt 👇",
]

HASHTAGS = ["#gaming", "#esports", "#gamer", "#clips", "#reels", "#twitch", "#latam"]

def openai_client():
    if not OpenAI:
        return None
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def build_caption_spicy(context: str) -> str:
    # si hay OpenAI -> mejor copy
    client = openai_client()
    if client:
        prompt = f"""Eres editor de clips gaming en español LATAM.
Objetivo: comentarios y debate (polémica sana).
Reglas:
- 1 HOOK fuerte (máx 8 palabras)
- 1 línea opinión con carácter (sin insultos)
- 1 pregunta que divida bandos
- Máx 35 palabras + 3-6 hashtags
Contexto: {context}
"""
        try:
            resp = client.responses.create(model=(OPENAI_MODEL or "gpt-4.1-mini"), input=prompt)
            txt = (getattr(resp, "output_text", "") or "").strip()
            if txt:
                return txt
        except Exception:
            pass

    # fallback templates
    a = random.choice(["skill", "suerte", "cringe", "cine"])
    b = random.choice(["trampa", "talento", "humo", "oro"])
    base = random.choice(CAPTION_TEMPLATES).format(a=a, b=b)
    tags = " ".join(random.sample(HASHTAGS, k=min(5, len(HASHTAGS))))
    return f"{base}\n{tags}"


# =========================
# IG publish
# =========================
def ig_api_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.post(url, data=data, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def ig_api_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def ig_wait_container(creation_id: str, access_token: str, timeout_sec: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        j = ig_api_get(f"{creation_id}", {"fields": "status_code", "access_token": access_token})
        status = (j.get("status_code") or "").upper()
        print("IG status:", status)
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {j}")
        time.sleep(3)
    raise TimeoutError(f"IG container not ready after {timeout_sec}s")

def ig_publish_reel(video_url: str, caption: str) -> Dict[str, Any]:
    if not (IG_USER_ID and IG_ACCESS_TOKEN):
        raise RuntimeError("Faltan IG_USER_ID o IG_ACCESS_TOKEN")
    print("Creating IG container")
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

    print("Waiting IG container:", creation_id)
    ig_wait_container(creation_id, IG_ACCESS_TOKEN, timeout_sec=900)

    print("Publishing IG")
    res = ig_api_post(f"{IG_USER_ID}/media_publish", {"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN})
    return res


# =========================
# Runway intro (image->video)
# =========================
def runway_headers() -> Dict[str, str]:
    if not RUNWAY_API_KEY:
        raise RuntimeError("Falta RUNWAY_API_KEY")
    return {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "X-Runway-Version": RUNWAY_VERSION,
        "Content-Type": "application/json",
    }

def runway_create_image_to_video(image_https_url: str, prompt_text: str, seconds: int) -> str:
    url = f"{RUNWAY_BASE}/v1/image_to_video"
    payload = {
        "model": RUNWAY_I2V_MODEL,
        "promptText": prompt_text[:900],
        "ratio": "720:1280",
        "duration": int(max(2, min(10, seconds))),
        "promptImage": image_https_url,
    }
    r = requests.post(url, headers=runway_headers(), json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    tid = j.get("id")
    if not tid:
        raise RuntimeError(f"Runway no devolvió id: {j}")
    return tid

def runway_get_task(task_id: str) -> Dict[str, Any]:
    url = f"{RUNWAY_BASE}/v1/tasks/{task_id}"
    r = requests.get(url, headers=runway_headers(), timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def runway_wait_for_mp4(task_id: str, timeout_sec: int) -> str:
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
            raise RuntimeError(f"Task OK pero sin mp4 en respuesta: {j}")
        if status in ("FAILED", "ERROR", "CANCELED", "CANCELLED"):
            raise RuntimeError(f"Runway task failed: {j}")
        time.sleep(RUNWAY_POLL_SEC)
    raise TimeoutError(f"Runway timeout. Last={last}")

def download_url_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.content


# =========================
# FFmpeg: gamer reel pipeline
# =========================
def extract_frame_as_jpg(video_path: str, out_jpg: str) -> None:
    # toma un frame temprano (0.5s)
    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-ss", "0.5", "-i", video_path, "-frames:v", "1", "-q:v", "3", out_jpg]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract falló: {(p.stderr or '')[:1200]}")

def build_filter(video_has_audio: bool, overlay_path: Optional[str]) -> str:
    # Base: crop para 9:16 + zoom lento + color + grain
    # Random: flash glitch + shake (aprox, sin volverse loco)
    zoom_expr = "zoom+0.00035"
    zoom_expr = f"if(lte(zoom,1.12),{zoom_expr},1.12)"
    base = (
        f"[0:v]"
        f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
        f"crop={REEL_W}:{REEL_H},fps={REEL_FPS},"
        f"zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={REEL_W}x{REEL_H},"
        f"eq=contrast={CONTRAST}:brightness={BRIGHT}:saturation={SAT},"
        f"noise=alls={GRAIN}:allf=t+u"
        f"[v0];"
    )

    # Glitch flash (simple): a veces invertimos/solarize con enable
    glitch_on = "between(t,2,2.08)+between(t,5,5.08)+between(t,8,8.08)"
    if random.random() > GLITCH_PROB:
        glitch_on = "0"

    glitch = (
        f"[v0]colorchannelmixer=rr='if({glitch_on},0.2,1)':gg='if({glitch_on},1.3,1)':bb='if({glitch_on},1.6,1)'"
        f"[v1];"
    )

    # Shake leve
    shake_on = "between(t,3,3.12)+between(t,6,6.12)"
    if random.random() > SHAKE_PROB:
        shake_on = "0"

    shake = (
        f"[v1]crop={REEL_W}:{REEL_H}:x='if({shake_on}, 6*sin(80*t), 0)':y='if({shake_on}, 6*cos(70*t), 0)'"
        f"[v2];"
    )

    if overlay_path:
        # overlay HUD con alpha
        ov = (
            f"[1:v]scale={REEL_W}:{REEL_H},format=rgba[ov];"
            f"[v2][ov]overlay=0:0:format=auto[vout]"
        )
        return base + glitch + shake + ov

    return base + glitch + shake + "[v2]copy[vout]"

def make_reel(input_mp4: str, output_mp4: str, overlay_png: Optional[str], music_mp3: Optional[str]) -> None:
    has_audio = ffprobe_has_audio(input_mp4)

    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", input_mp4]

    if overlay_png:
        cmd += ["-i", overlay_png]
    if music_mp3:
        cmd += ["-i", music_mp3]

    # filtros video
    filt = build_filter(has_audio, overlay_png)

    cmd += ["-filter_complex", filt, "-map", "[vout]"]

    # audio mix
    if has_audio and music_mp3:
        # inputs:
        # 0 = video (con audio)
        # 1 = overlay (si existe)
        # last = music
        music_idx = 2 if overlay_png else 1
        cmd += [
            "-map", "0:a:0",
            "-map", f"{music_idx}:a:0",
            "-filter_complex",
            filt + f";[0:a]volume={ORIG_VOLUME}[a0];[{music_idx}:a]volume={MUSIC_VOLUME}[a1];[a0][a1]amix=inputs=2:duration=shortest:dropout_transition=2[aout]",
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:a", "aac", "-b:a", "128k"
        ]
    elif has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"]
    elif music_mp3:
        music_idx = 2 if overlay_png else 1
        cmd += ["-map", f"{music_idx}:a:0", "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]

    cmd += [
        "-t", str(int(REEL_SECONDS)),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4
    ]

    # importante: como usamos -filter_complex dos veces en mix, hacemos una versión simple:
    # => si hay mix, el cmd arriba duplicaría filter_complex.
    # Para no enredar: si hay mix, hacemos otra ruta.
    if has_audio and music_mp3:
        cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-i", input_mp4]
        if overlay_png:
            cmd += ["-i", overlay_png]
        cmd += ["-i", music_mp3]

        music_idx = 2 if overlay_png else 1
        filt2 = build_filter(has_audio, overlay_png)

        cmd += ["-filter_complex",
                filt2 +
                f";[0:a]volume={ORIG_VOLUME}[a0];[{music_idx}:a]volume={MUSIC_VOLUME}[a1];"
                f"[a0][a1]amix=inputs=2:duration=shortest:dropout_transition=2[aout]",
                "-map", "[vout]", "-map", "[aout]",
                "-t", str(int(REEL_SECONDS)),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                output_mp4
                ]

    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg falló:\n{(p.stderr or '')[:4000]}")

def concat_intro(intro_mp4: str, main_mp4: str, out_mp4: str) -> None:
    # concat demuxer (requiere mismos codecs; aquí ambos x264/aac)
    with tempfile.TemporaryDirectory() as td:
        lst = os.path.join(td, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write(f"file '{intro_mp4}'\n")
            f.write(f"file '{main_mp4}'\n")
        cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", lst,
               "-c", "copy", out_mp4]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        if p.returncode != 0:
            # fallback: re-encode
            cmd2 = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", intro_mp4, "-i", main_mp4,
                    "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    out_mp4]
            p2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600, check=False)
            if p2.returncode != 0:
                raise RuntimeError(f"concat falló:\n{(p2.stderr or '')[:2500]}")


# =========================
# Main Mode B
# =========================
def run_mode_b() -> None:
    print("UGC MODE B START")

    keys = list_r2_keys(INBOX_PREFIX + "/")
    if not keys:
        print("Inbox vacío ✅")
        return

    # random sorpresa: baraja orden cada corrida
    random.shuffle(keys)

    overlays = list_hud_overlays()
    music = pick_music_file()

    for key in keys:
        print("Processing:", key)
        data = get_r2_bytes(key)
        h = safe_hash(data)

        with tempfile.TemporaryDirectory() as td:
            in_mp4 = os.path.join(td, "in.mp4")
            out_main = os.path.join(td, "main.mp4")
            out_final = os.path.join(td, "final.mp4")
            frame_jpg = os.path.join(td, "frame.jpg")

            with open(in_mp4, "wb") as f:
                f.write(data)

            overlay_png = random.choice(overlays) if overlays else None

            # 1) main reel con estilo gamer
            make_reel(in_mp4, out_main, overlay_png=overlay_png, music_mp3=music)

            # 2) Runway intro sorpresa (opcional)
            use_intro = (
                RUNWAY_ENABLED and bool(RUNWAY_API_KEY) and bool(RUNWAY_BASE)
                and (random.random() <= max(0.0, min(1.0, RUNWAY_INTRO_PROB)))
            )

            if use_intro:
                try:
                    extract_frame_as_jpg(in_mp4, frame_jpg)
                    with open(frame_jpg, "rb") as f:
                        img_bytes = f.read()

                    # subimos frame a R2 para darle URL https a Runway
                    img_key = f"{OUT_PREFIX}/runway_frames/{h}.jpg"
                    img_url = put_r2_bytes(img_key, img_bytes, content_type="image/jpeg")

                    prompt = "Esports neon glitch cinematic intro, HUD overlays, high energy, cyberpunk gaming vibe"
                    task_id = runway_create_image_to_video(img_url, prompt, seconds=RUNWAY_INTRO_SECONDS)
                    print("Runway task:", task_id)

                    mp4_url = runway_wait_for_mp4(task_id, timeout_sec=RUNWAY_TIMEOUT)
                    intro_bytes = download_url_bytes(mp4_url)

                    intro_mp4 = os.path.join(td, "intro.mp4")
                    with open(intro_mp4, "wb") as f:
                        f.write(intro_bytes)

                    # convert intro a 1080x1920 con audio mudo (para concat fácil)
                    intro_norm = os.path.join(td, "intro_norm.mp4")
                    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                           "-i", intro_mp4,
                           "-vf", f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,crop={REEL_W}:{REEL_H},fps={REEL_FPS}",
                           "-an",
                           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart",
                           intro_norm]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)

                    # añade audio silent para concat v+a más robusto
                    intro_a = os.path.join(td, "intro_a.mp4")
                    cmd2 = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                            "-i", intro_norm,
                            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                            "-shortest",
                            "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "128k",
                            intro_a]
                    subprocess.run(cmd2, capture_output=True, text=True, timeout=300, check=True)

                    # main ya tiene o no audio, forzamos audio silent si no tiene
                    main_a = os.path.join(td, "main_a.mp4")
                    if ffprobe_has_audio(out_main):
                        # ok
                        main_a = out_main
                    else:
                        cmd3 = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                                "-i", out_main,
                                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                                "-shortest",
                                "-c:v", "copy",
                                "-c:a", "aac", "-b:a", "128k",
                                main_a]
                        subprocess.run(cmd3, capture_output=True, text=True, timeout=300, check=True)

                    concat_intro(intro_a, main_a, out_final)
                    final_path = out_final
                    print("Runway intro ✅")
                except Exception as e:
                    print("Runway intro falló (sigue sin romper):", str(e))
                    final_path = out_main
            else:
                final_path = out_main

            with open(final_path, "rb") as f:
                final_bytes = f.read()

        out_key = f"{OUT_PREFIX}/{h}.mp4"
        public_url = put_r2_bytes(out_key, final_bytes, content_type="video/mp4")

        # Caption
        context = f"Clip gaming procesado desde {key.split('/')[-1]}"
        caption = build_caption_spicy(context)
        print("CAPTION:", caption)

        # IG publish
        if ENABLE_IG_PUBLISH:
            try:
                res = ig_publish_reel(public_url, caption)
                print("IG publish:", res)
            except Exception as e:
                print("IG publish falló (no rompe):", str(e))

        print("Publicado:", public_url)

        # ✅ Opcional: mover o borrar inbox para que no se reprocesse
        # Yo recomiendo BORRAR de inbox al procesar:
        try:
            delete_r2_key(key)
        except Exception as e:
            print("No pude borrar inbox (no rompe):", str(e))

    print("UGC MODE B DONE")
