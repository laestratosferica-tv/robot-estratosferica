# ugc_mode_b.py
import os
import re
import json
import time
import random
import hashlib
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import boto3
import requests

# -------------------------
# Helpers env
# -------------------------

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

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# -------------------------
# Core env (R2/IG/OpenAI)
# -------------------------

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL") or "").rstrip("/")

# Instagram
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", True)
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
IG_USER_ID = env_nonempty("IG_USER_ID")
GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

# ✅ DRY RUN: simula todo, NO publica en IG
UGC_DRY_RUN = env_bool("UGC_DRY_RUN", False)

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

# UGC folders in R2
UGC_INBOX_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox/") or "ugc/inbox/").strip().lstrip("/")
UGC_LIBRARY_PREFIX = (env_nonempty("UGC_LIBRARY_PREFIX", "ugc/library/raw/") or "ugc/library/raw/").strip().lstrip("/")
UGC_PROCESSED_PREFIX = (env_nonempty("UGC_PROCESSED_PREFIX", "ugc/processed/") or "ugc/processed/").strip().lstrip("/")
UGC_FAILED_PREFIX = (env_nonempty("UGC_FAILED_PREFIX", "ugc/failed/") or "ugc/failed/").strip().lstrip("/")

UGC_QUEUE_PENDING = (env_nonempty("UGC_QUEUE_PENDING", "ugc/queue/pending/") or "ugc/queue/pending/").strip().lstrip("/")
UGC_QUEUE_PUBLISHED = (env_nonempty("UGC_QUEUE_PUBLISHED", "ugc/queue/published/") or "ugc/queue/published/").strip().lstrip("/")
UGC_QUEUE_FAILED = (env_nonempty("UGC_QUEUE_FAILED", "ugc/queue/failed/") or "ugc/queue/failed/").strip().lstrip("/")

UGC_OUTPUT_REELS_PREFIX = (env_nonempty("UGC_OUTPUT_REELS_PREFIX", "ugc/outputs/reels/") or "ugc/outputs/reels/").strip().lstrip("/")
UGC_OUTPUT_AUDIO_PREFIX = (env_nonempty("UGC_OUTPUT_AUDIO_PREFIX", "ugc/outputs/audio/") or "ugc/outputs/audio/").strip().lstrip("/")

UGC_STATE_KEY = (env_nonempty("UGC_STATE_KEY", "ugc/state/state.json") or "ugc/state/state.json").strip().lstrip("/")

# Publish cadence
MAX_POSTS_PER_DAY = env_int("MAX_POSTS_PER_DAY", 1)
LOCAL_TZ = ZoneInfo("America/Bogota")

# Video rules
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
UGC_CAP_SECONDS = env_int("UGC_CAP_SECONDS", 30)
UGC_MIN_SECONDS = env_int("UGC_MIN_SECONDS", 5)
UGC_FFMPEG_TIMEOUT = env_int("UGC_FFMPEG_TIMEOUT", 900)

# Voice & music roulette (normal modes)
ROULETTE_TEXT_PCT = env_int("ROULETTE_TEXT_PCT", 20)
ROULETTE_MUSIC_PCT = env_int("ROULETTE_MUSIC_PCT", 25)
ROULETTE_VOICE_PCT = env_int("ROULETTE_VOICE_PCT", 25)
ROULETTE_VOICE_MUSIC_PCT = env_int("ROULETTE_VOICE_MUSIC_PCT", 30)

# Rare events
RARE_EVENT_PCT = env_int("RARE_EVENT_PCT", 5)
RARE_EVENT_LABEL = env_nonempty("RARE_EVENT_LABEL", "🛸 MODO RARO") or "🛸 MODO RARO"
RARE_EVENT_VOICE = env_nonempty("RARE_EVENT_VOICE", "echo") or "echo"
RARE_EVENT_VOICE_INSTR = env_nonempty(
    "RARE_EVENT_VOICE_INSTR",
    "Voz estilo alien/robot, metálica, misteriosa, con hype gamer. Clara y entendible."
) or "Voz estilo alien/robot, metálica, misteriosa, con hype gamer. Clara y entendible."

NCS_RARE_SLUGS = [
    "mortals",
    "heroes-tonight",
    "invisible",
    "symbolism",
]

# Audio mix volumes
MUSIC_VOLUME = env_float("MUSIC_VOLUME", 0.12)
VOICE_VOLUME = env_float("VOICE_VOLUME", 1.0)
ORIG_VOLUME = env_float("ORIG_VOLUME", 0.18)

# OpenAI
OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL_TEXT = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TTS_MODEL = env_nonempty("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

# NCS slugs (normal pool)
NCS_TRACK_SLUGS = [
    "montagemindia",
    "favela",
    "numb",
    "skyhigh",
    "mortals",
    "invisible",
    "heroes-tonight",
    "symbolism",
]

VOICE_PRESETS = [
    {"voice": "nova", "instr": "Voz femenina LATAM, caster esports, rápida, hype, divertida."},
    {"voice": "onyx", "instr": "Voz masculina LATAM, narrador épico tipo tráiler, intensa y dramática."},
    {"voice": "shimmer", "instr": "Voz femenina, meme/humor gamer, sarcástica pero amable."},
    {"voice": "alloy", "instr": "Voz neutra, análisis competitivo, segura y clara."},
    {"voice": "echo", "instr": "Voz estilo 'alien/robot', metálica, misteriosa, pero entendible."},
]

# -------------------------
# R2 client helpers
# -------------------------

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

def s3_exists(key: str) -> bool:
    try:
        r2_client().head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False

def s3_list_keys(prefix: str, max_keys: int = 200) -> List[str]:
    s3 = r2_client()
    out: List[str] = []
    token = None
    for _ in range(10):
        kwargs = {"Bucket": BUCKET_NAME, "Prefix": prefix, "MaxKeys": max_keys}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key")
            if k and not k.endswith("/"):
                out.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out

def s3_get_bytes(key: str) -> bytes:
    s3 = r2_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return obj["Body"].read()

def s3_put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3 = r2_client()
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)

def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3_put_bytes(key, b, "application/json")

def s3_get_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        b = s3_get_bytes(key)
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None

def s3_copy_delete(src_key: str, dst_key: str) -> None:
    s3 = r2_client()
    s3.copy_object(Bucket=BUCKET_NAME, CopySource={"Bucket": BUCKET_NAME, "Key": src_key}, Key=dst_key)
    s3.delete_object(Bucket=BUCKET_NAME, Key=src_key)

def r2_public_url(key: str) -> str:
    base = (env_nonempty("R2_PUBLIC_BASE_URL", R2_PUBLIC_BASE_URL) or "").rstrip("/")
    if not base.startswith("http"):
        raise RuntimeError("R2_PUBLIC_BASE_URL inválido o vacío (debe empezar por https://)")
    return f"{base}/{key}"


# -------------------------
# Instagram Graph API
# -------------------------

def _raise_meta_error(r: requests.Response, label: str) -> None:
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

def ig_wait_container(creation_id: str, timeout_sec: int = 420) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        j = ig_api_get(
            f"{creation_id}",
            {"fields": "status_code", "access_token": IG_ACCESS_TOKEN},
        )
        status = (j.get("status_code") or "").upper()
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {j}")
        time.sleep(3)
    raise TimeoutError(f"IG container not ready after {timeout_sec}s")

def ig_publish_reel(video_url: str, caption: str) -> Dict[str, Any]:
    # ✅ DRY RUN: no publica, solo imprime
    if UGC_DRY_RUN:
        print("[UGC][DRY_RUN] NO publico en IG.")
        print("[UGC][DRY_RUN] video_url:", video_url)
        print("[UGC][DRY_RUN] caption preview:", (caption or "")[:300])
        return {"dry_run": True, "video_url": video_url, "caption_preview": (caption or "")[:300]}

    if not (IG_USER_ID and IG_ACCESS_TOKEN):
        raise RuntimeError("Faltan IG_USER_ID o IG_ACCESS_TOKEN")

    j = ig_api_post(
        f"{IG_USER_ID}/media",
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": IG_ACCESS_TOKEN,
        },
    )
    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG reels create failed: {j}")
    ig_wait_container(creation_id)
    res = ig_api_post(
        f"{IG_USER_ID}/media_publish",
        {"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN},
    )
    return res


# -------------------------
# OpenAI (text) + TTS via HTTP
# -------------------------

def openai_text(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")
    model = OPENAI_MODEL_TEXT or "gpt-4.1-mini"
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
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

    try:
        chunks = j.get("output", [])
        texts = []
        for c in chunks:
            for part in c.get("content", []) or []:
                if part.get("type") == "output_text" and part.get("text"):
                    texts.append(part["text"])
        return ("\n".join(texts)).strip()
    except Exception:
        return ""

def openai_tts_mp3(text: str, voice: str, instructions: str) -> bytes:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")
    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_TTS_MODEL,
        "voice": voice,
        "format": "mp3",
        "input": text,
        "instructions": instructions,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content


# -------------------------
# Video / audio processing
# -------------------------

def ffprobe_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if p.returncode != 0:
        return 0.0
    try:
        return float((p.stdout or "").strip())
    except Exception:
        return 0.0

def ffprobe_has_audio(path: str) -> bool:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=nw=1:nk=1",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if p.returncode != 0:
        return False
    return bool((p.stdout or "").strip())

def make_reel_video(
    in_mp4: str,
    out_mp4: str,
    target_seconds: int,
    music_mp3: Optional[str],
    voice_mp3: Optional[str],
) -> None:
    vf = (
        f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
        f"crop={REEL_W}:{REEL_H},"
        f"fps=30"
    )

    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", in_mp4]

    audio_inputs = []
    if music_mp3 and os.path.exists(music_mp3):
        cmd += ["-i", music_mp3]
        audio_inputs.append(("music", len(audio_inputs) + 1))
    if voice_mp3 and os.path.exists(voice_mp3):
        cmd += ["-i", voice_mp3]
        audio_inputs.append(("voice", len(audio_inputs) + 1))

    filter_complex_parts = [f"[0:v]{vf}[vout]"]
    has_external = bool(audio_inputs)

    if has_external:
        has_audio = ffprobe_has_audio(in_mp4)
        parts = []
        if has_audio:
            parts.append(f"[0:a]volume={ORIG_VOLUME}[a0]")
            base_audio = "[a0]"
        else:
            parts.append("anullsrc=channel_layout=stereo:sample_rate=44100,volume=0.0[a0]")
            base_audio = "[a0]"

        idx = 1
        for name, _rel in audio_inputs:
            label = "m" if name == "music" else "vo"
            vol = MUSIC_VOLUME if name == "music" else VOICE_VOLUME
            parts.append(f"[{idx}:a]volume={vol}[a{label}]")
            idx += 1

        filter_complex_parts += parts

        mix_inputs = [base_audio]
        if any(n == "music" for n, _ in audio_inputs):
            mix_inputs.append("[am]")
        if any(n == "voice" for n, _ in audio_inputs):
            mix_inputs.append("[avo]")

        amix = "".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=2[aout]"
        filter_complex_parts.append(amix)
        map_audio = ["-map", "[aout]"]
        audio_codec = ["-c:a", "aac", "-b:a", "128k"]
    else:
        map_audio = ["-map", "0:a?"]
        audio_codec = ["-c:a", "aac", "-b:a", "128k"]

    filter_complex = ";".join(filter_complex_parts)

    cmd += [
        "-t", str(int(target_seconds)),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        *map_audio,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        *audio_codec,
        "-shortest",
        out_mp4,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, timeout=UGC_FFMPEG_TIMEOUT, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{(p.stderr or '')[:4000]}")


# -------------------------
# NCS music downloader (best-effort)
# -------------------------

NCS_MP3_RE = re.compile(r'https?://[^\s"\']+\.mp3', re.IGNORECASE)

def try_download_ncs_mp3(slug: str) -> Optional[Tuple[bytes, str]]:
    try:
        url = f"https://ncs.io/{slug}"
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code >= 400:
            return None
        html = r.text
        m = NCS_MP3_RE.search(html)
        if not m:
            return None
        mp3_url = m.group(0)
        rr = requests.get(mp3_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if rr.status_code >= 400 or not rr.content:
            return None
        credit = f"Music: NCS (NoCopyrightSounds) – https://ncs.io/{slug}"
        return rr.content, credit
    except Exception:
        return None


# -------------------------
# Roulette deterministic + caption/narration
# -------------------------

def _stable_seed_from_key(key: str) -> int:
    h = hashlib.sha1((key or "").encode("utf-8")).hexdigest()[:8]
    return int(h, 16)

def roulette_pick_mode_rng(rng: random.Random) -> str:
    if RARE_EVENT_PCT > 0:
        roll = rng.randint(1, 100)
        if roll <= RARE_EVENT_PCT:
            return "rare_event"

    modes = (
        ["text_only"] * max(0, ROULETTE_TEXT_PCT) +
        ["music_only"] * max(0, ROULETTE_MUSIC_PCT) +
        ["voice_only"] * max(0, ROULETTE_VOICE_PCT) +
        ["voice_music"] * max(0, ROULETTE_VOICE_MUSIC_PCT)
    )
    return rng.choice(modes) if modes else "voice_music"

def choose_voice_preset_rng(rng: random.Random) -> Dict[str, str]:
    return rng.choice(VOICE_PRESETS)

def choose_ncs_slug_rng(rng: random.Random, *, rare: bool = False) -> str:
    if rare and NCS_RARE_SLUGS:
        return rng.choice(NCS_RARE_SLUGS)
    return rng.choice(NCS_TRACK_SLUGS)

def build_caption_and_narration(filename: str, mode: str) -> Tuple[str, str]:
    prompt = f"""
Eres editor viral de Reels esports/gaming en español LATAM.
Contexto: El usuario subió un video (probablemente gameplay/clip) llamado: "{filename}".

Objetivo: maximizar retención y comentarios. No inventes marcas ni derechos.

Entrega 2 cosas:
1) CAPTION: 1-2 párrafos cortos + 6-12 hashtags relevantes al final + termina con una pregunta.
2) NARRATION (voz): 1 guion de 1 a 2 frases MUY cortas estilo caster/narrador. Hook fuerte en la primera frase.

Modo elegido: {mode}
Si modo = text_only o music_only, la narración puede ser vacía o mínima.
Si modo = rare_event, la narración debe ser MUY corta pero MUY hype (suena a alien/robot).

Formato EXACTO:
CAPTION:
...
NARRATION:
...
"""
    out = openai_text(prompt)
    caption = ""
    narration = ""
    if "CAPTION:" in out:
        parts = out.split("CAPTION:", 1)[1]
        if "NARRATION:" in parts:
            cap, nar = parts.split("NARRATION:", 1)
            caption = cap.strip()
            narration = nar.strip()
        else:
            caption = parts.strip()
    else:
        caption = out.strip()

    caption = caption[:2200].strip()
    narration = narration[:220].strip()
    return caption, narration


# -------------------------
# Queue/state
# -------------------------

def load_state() -> Dict[str, Any]:
    st = s3_get_json(UGC_STATE_KEY) or {}
    st.setdefault("daily", {})
    st.setdefault("enqueued", {})
    return st

def save_state(st: Dict[str, Any]) -> None:
    s3_put_json(UGC_STATE_KEY, st)

def today_key_local() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")

def daily_count(st: Dict[str, Any], day_key: str) -> int:
    return int((st.get("daily", {}).get(day_key) or {}).get("published", 0))

def daily_inc(st: Dict[str, Any], day_key: str) -> None:
    st.setdefault("daily", {}).setdefault(day_key, {})
    st["daily"][day_key]["published"] = int(st["daily"][day_key].get("published", 0)) + 1
    st["daily"][day_key]["last_published_at"] = iso_now()

def enqueue_new_inbox_videos(st: Dict[str, Any]) -> int:
    keys = s3_list_keys(UGC_INBOX_PREFIX)
    mp4s = [k for k in keys if k.lower().endswith(".mp4")]
    created = 0

    for k in sorted(mp4s):
        if k in st["enqueued"]:
            continue

        filename = k.split("/")[-1]
        rng = random.Random(_stable_seed_from_key(k))

        mode = roulette_pick_mode_rng(rng)
        is_rare = (mode == "rare_event")

        voice = None
        voice_instr = None
        if is_rare:
            voice = RARE_EVENT_VOICE
            voice_instr = RARE_EVENT_VOICE_INSTR
        elif mode in ("voice_only", "voice_music"):
            vp = choose_voice_preset_rng(rng)
            voice = vp["voice"]
            voice_instr = vp["instr"]

        music_slug = None
        if is_rare:
            music_slug = choose_ncs_slug_rng(rng, rare=True)
        elif mode in ("music_only", "voice_music"):
            music_slug = choose_ncs_slug_rng(rng, rare=False)

        caption, narration = build_caption_and_narration(filename, mode)

        if is_rare:
            stamp = f"\n\n{RARE_EVENT_LABEL} (salió en ruleta)"
            caption = (caption + stamp)[:2200].strip()

        ticket = {
            "created_at": iso_now(),
            "source": {"inbox_key": k, "filename": filename},
            "roulette": {
                "mode": mode,
                "voice": voice,
                "voice_instructions": voice_instr,
                "music_ncs_slug": music_slug,
                "is_rare_event": is_rare,
            },
            "ai": {
                "caption": caption,
                "narration": narration,
            },
            "status": "pending",
        }

        ticket_key = f"{UGC_QUEUE_PENDING}{today_key_local()}__{short_hash(k + iso_now())}__{filename}.json"
        s3_put_json(ticket_key, ticket)
        st["enqueued"][k] = ticket_key
        created += 1

        print(f"[UGC] Encolado: {k} -> {ticket_key}")

    return created

def list_pending_tickets() -> List[str]:
    keys = s3_list_keys(UGC_QUEUE_PENDING)
    return sorted([k for k in keys if k.lower().endswith(".json")])

def mark_ticket_move(ticket_key: str, dst_prefix: str, patch: Dict[str, Any]) -> str:
    data = s3_get_json(ticket_key) or {}
    for kk, vv in patch.items():
        data[kk] = vv
    data["updated_at"] = iso_now()

    fname = ticket_key.split("/")[-1]
    new_key = f"{dst_prefix}{fname}"
    s3_put_json(new_key, data)
    r2_client().delete_object(Bucket=BUCKET_NAME, Key=ticket_key)
    return new_key


# -------------------------
# Main publish worker
# -------------------------

def publish_one_from_queue_if_allowed(st: Dict[str, Any]) -> int:
    # ✅ En DRY RUN seguimos aunque ENABLE_IG_PUBLISH sea false; pero tú lo dejarás true.
    if not ENABLE_IG_PUBLISH and not UGC_DRY_RUN:
        print("[UGC] ENABLE_IG_PUBLISH=false, no publico.")
        return 0

    if not (IG_USER_ID and IG_ACCESS_TOKEN) and not UGC_DRY_RUN:
        print("[UGC] Faltan IG_USER_ID / IG_ACCESS_TOKEN, no publico.")
        return 0

    day = today_key_local()
    used = daily_count(st, day)
    if used >= MAX_POSTS_PER_DAY:
        print(f"[UGC] Cupo diario alcanzado ({used}/{MAX_POSTS_PER_DAY}). No publico más hoy.")
        return 0

    pending = list_pending_tickets()
    if not pending:
        print("[UGC] No hay tickets en pending.")
        return 0

    max_attempts = min(5, len(pending))
    for i in range(max_attempts):
        ticket_key = pending[i]
        ticket = s3_get_json(ticket_key) or {}

        src_key = (((ticket.get("source") or {}).get("inbox_key")) or "").strip()
        filename = ((ticket.get("source") or {}).get("filename") or "ugc.mp4")

        if not src_key:
            err = "Ticket sin source.inbox_key (src_key vacío)"
            print("[UGC] Ticket inválido:", ticket_key, "|", err)
            try:
                mark_ticket_move(ticket_key, UGC_QUEUE_FAILED, {"status": "failed", "error": err})
            except Exception as ee:
                print("[UGC] Además falló mover ticket inválido a failed:", str(ee))
            continue

        roulette = ticket.get("roulette") or {}
        mode = roulette.get("mode") or "voice_music"
        voice = roulette.get("voice")
        voice_instr = roulette.get("voice_instructions") or ""
        music_slug = roulette.get("music_ncs_slug")

        caption = ((ticket.get("ai") or {}).get("caption") or "").strip()
        narration = ((ticket.get("ai") or {}).get("narration") or "").strip()

        print(f"[UGC] Intento {i+1}/{max_attempts} | Ticket: {ticket_key}")
        print(f"[UGC] Mode: {mode} | voice={voice} | music_slug={music_slug}")

        try:
            video_bytes = s3_get_bytes(src_key)
            with tempfile.TemporaryDirectory() as td:
                in_mp4 = os.path.join(td, "in.mp4")
                with open(in_mp4, "wb") as f:
                    f.write(video_bytes)

                dur = ffprobe_duration_seconds(in_mp4)
                if dur <= 0:
                    dur = float(UGC_MIN_SECONDS)

                target = int(min(max(dur, float(UGC_MIN_SECONDS)), float(UGC_CAP_SECONDS)))

                # Optional music
                music_path = None
                music_credit = ""
                if mode in ("music_only", "voice_music", "rare_event") and music_slug:
                    got = try_download_ncs_mp3(music_slug)
                    if got:
                        music_bytes, music_credit = got
                        music_path = os.path.join(td, "music.mp3")
                        with open(music_path, "wb") as f:
                            f.write(music_bytes)
                    else:
                        print("[UGC] Música NCS no disponible (best-effort). Sigo sin música.")

                # Optional voice
                voice_path = None
                if mode in ("voice_only", "voice_music", "rare_event") and narration and voice:
                    try:
                        tts_bytes = openai_tts_mp3(narration, voice=voice, instructions=voice_instr)
                        voice_path = os.path.join(td, "voice.mp3")
                        with open(voice_path, "wb") as f:
                            f.write(tts_bytes)
                    except Exception as e:
                        print("[UGC] TTS falló (no rompe). Sigo sin voz. Error:", str(e))
                        voice_path = None

                out_mp4 = os.path.join(td, "reel.mp4")
                make_reel_video(
                    in_mp4=in_mp4,
                    out_mp4=out_mp4,
                    target_seconds=target,
                    music_mp3=music_path if mode in ("music_only", "voice_music", "rare_event") else None,
                    voice_mp3=voice_path if mode in ("voice_only", "voice_music", "rare_event") else None,
                )

                out_bytes = open(out_mp4, "rb").read()
                reel_key = f"{UGC_OUTPUT_REELS_PREFIX}{today_key_local()}__{short_hash(src_key + iso_now())}__{filename}"
                s3_put_bytes(reel_key, out_bytes, "video/mp4")
                video_url = r2_public_url(reel_key)
                print("[UGC] REEL subido a R2:", video_url)

                final_caption = caption
                if music_credit:
                    final_caption = (final_caption + "\n\n" + music_credit).strip()

                # ✅ Aquí se simula (no publica) si UGC_DRY_RUN=true
                res = ig_publish_reel(video_url=video_url, caption=final_caption)

                # Move ticket to published (sí, incluso en dry run, para simular el flujo)
                mark_ticket_move(
                    ticket_key,
                    UGC_QUEUE_PUBLISHED,
                    {
                        "status": "published",
                        "publish": {
                            "ig": res,
                            "video_url": video_url,
                            "published_at": iso_now(),
                            "dry_run": bool(UGC_DRY_RUN),
                        },
                        "render": {"reel_key": reel_key, "target_seconds": target},
                    },
                )

                # Move original out of inbox -> library/raw (master) and COPY to processed
                lib_key = f"{UGC_LIBRARY_PREFIX}{filename}"
                s3_copy_delete(src_key, lib_key)

                proc_key = f"{UGC_PROCESSED_PREFIX}{filename}"
                r2_client().copy_object(
                    Bucket=BUCKET_NAME,
                    CopySource={"Bucket": BUCKET_NAME, "Key": lib_key},
                    Key=proc_key,
                )
                print("[UGC] Original guardado en library/raw y copiado a processed:", proc_key)

                daily_inc(st, day)
                save_state(st)
                return 1

        except Exception as e:
            err = str(e)
            print("[UGC] FALLÓ este ticket:", ticket_key, "| Error:", err)

            try:
                mark_ticket_move(ticket_key, UGC_QUEUE_FAILED, {"status": "failed", "error": err})
                print("[UGC] Ticket movido a failed.")
            except Exception as ee:
                print("[UGC] Además falló mover ticket a failed:", str(ee))

            try:
                if src_key.startswith(UGC_INBOX_PREFIX) and s3_exists(src_key):
                    fail_name = f"{today_key_local()}__{short_hash(src_key + iso_now())}__{filename}"
                    failed_video_key = f"{UGC_FAILED_PREFIX}{fail_name}"
                    s3_copy_delete(src_key, failed_video_key)
                    print("[UGC] Video original movido a failed:", failed_video_key)
                else:
                    print("[UGC] Video original no movido (no está en inbox o no existe):", src_key)
            except Exception as ee:
                print("[UGC] Falló mover video original a failed (no rompe):", str(ee))

            continue

    print("[UGC] No se pudo publicar ningún ticket en este run (todos fallaron o eran inválidos).")
    return 0


# -------------------------
# Entrypoint
# -------------------------

def run_mode_b() -> None:
    print("===== MODO B (UGC QUEUE + 1/DÍA + RULETA + NCS best-effort) =====")
    st = load_state()

    created = enqueue_new_inbox_videos(st)
    if created:
        save_state(st)
    print(f"[UGC] Tickets nuevos creados: {created}")

    published = publish_one_from_queue_if_allowed(st)
    print(f"[UGC] Publicados en este run: {published}")

    day = today_key_local()
    print(f"[UGC] Hoy llevas: {daily_count(st, day)}/{MAX_POSTS_PER_DAY} publicados (TZ America/Bogota)")
    print("[UGC] Listo.")
