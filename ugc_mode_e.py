# ugc_mode_e.py
import os
import re
import json
import time
import random
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import boto3


# =========================
# ENV helpers
# =========================

def env_nonempty(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default

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
# Twitch
# =========================

TWITCH_CLIENT_ID = env_nonempty("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = env_nonempty("TWITCH_CLIENT_SECRET")

TWITCH_GAMES = env_nonempty(
    "TWITCH_GAMES",
    "Just Chatting,League of Legends,Counter-Strike,Grand Theft Auto V,Minecraft,VALORANT,Apex Legends"
)

TWITCH_CLIPS_PER_GAME = env_int("TWITCH_CLIPS_PER_GAME", 6)
TWITCH_PICK_TOTAL = env_int("TWITCH_PICK_TOTAL", 3)
TWITCH_MIN_VIEWS = env_int("TWITCH_MIN_VIEWS", 800)
TWITCH_PERIOD_DAYS = env_int("TWITCH_PERIOD_DAYS", 7)

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

# =========================
# R2
# =========================
AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or "").rstrip("/")
R2_INBOX_PREFIX = (env_nonempty("R2_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")

USER_AGENT = "Mozilla/5.0 (robot-ugc-e)"

# Twitch clip thumbnails: ...-preview-480x272.jpg  -> mp4 real: ... .mp4
CLIP_PREVIEW_RE = re.compile(r"-preview-\d+x\d+\.jpg($|\?)", re.IGNORECASE)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_now() -> str:
    return now_utc().strftime("%Y-%m-%d")

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

def upload_bytes_to_r2(key: str, data: bytes, content_type: str = "video/mp4") -> str:
    s3 = r2_client()
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    if R2_PUBLIC_BASE_URL.startswith("http"):
        return f"{R2_PUBLIC_BASE_URL}/{key}"
    return key

def twitch_app_token() -> str:
    if not (TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET):
        raise RuntimeError("Faltan env Twitch: TWITCH_CLIENT_ID y TWITCH_CLIENT_SECRET")

    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": TWITCH_CLIENT_ID,
        "client_secret": TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    r = requests.post(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    tok = j.get("access_token")
    if not tok:
        raise RuntimeError(f"Twitch no devolvió access_token: {j}")
    return tok

def twitch_headers(token: str) -> Dict[str, str]:
    return {
        "Client-ID": TWITCH_CLIENT_ID or "",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }

def twitch_get_game_id(token: str, game_name: str) -> Optional[str]:
    url = "https://api.twitch.tv/helix/games"
    r = requests.get(url, headers=twitch_headers(token), params={"name": game_name}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = (r.json() or {}).get("data") or []
    if not data:
        return None
    return data[0].get("id")

def twitch_get_clips(token: str, game_id: str, first: int, started_at_iso: str) -> List[Dict[str, Any]]:
    url = "https://api.twitch.tv/helix/clips"
    params = {
        "game_id": game_id,
        "first": int(max(1, min(100, first))),
        "started_at": started_at_iso,
    }
    r = requests.get(url, headers=twitch_headers(token), params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return (r.json() or {}).get("data") or []

def clip_mp4_from_thumbnail(thumbnail_url: str) -> Optional[str]:
    """
    Clips reales usan thumbnails tipo:
      https://clips-media-assets2.twitch.tv/...-preview-480x272.jpg
    El mp4 real suele ser:
      https://clips-media-assets2.twitch.tv/....mp4
    """
    if not thumbnail_url:
        return None
    u = thumbnail_url.strip()
    if not u.startswith("http"):
        return None
    if CLIP_PREVIEW_RE.search(u):
        u2 = CLIP_PREVIEW_RE.sub(".mp4", u)
        return u2.split("?")[0]
    # fallback: si ya viene mp4
    if u.lower().endswith(".mp4"):
        return u
    return None

def download_bytes(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r.content

def safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9\-_]+", "_", s)[:120]
    return s.strip("_") or "clip"

def run_mode_e() -> None:
    print("===== MODE E (TWITCH) -> R2 INBOX =====")

    token = twitch_app_token()

    # periodo
    started_at = (now_utc() - timedelta_days(TWITCH_PERIOD_DAYS)).isoformat().replace("+00:00", "Z")

    game_names = [x.strip() for x in (TWITCH_GAMES or "").split(",") if x.strip()]
    if not game_names:
        raise RuntimeError("TWITCH_GAMES vacío. Pon algo tipo: 'VALORANT,League of Legends,Counter-Strike'")

    all_candidates: List[Dict[str, Any]] = []

    for g in game_names:
        print("Exploring:", g)
        gid = twitch_get_game_id(token, g)
        if not gid:
            print(" - No game_id encontrado, skip")
            continue

        clips = twitch_get_clips(token, gid, first=TWITCH_CLIPS_PER_GAME, started_at_iso=started_at)
        for c in clips:
            views = int(c.get("view_count") or 0)
            thumb = c.get("thumbnail_url") or ""
            mp4 = clip_mp4_from_thumbnail(thumb)

            if not mp4:
                continue
            if views < TWITCH_MIN_VIEWS:
                continue

            all_candidates.append({
                "game": g,
                "id": c.get("id"),
                "title": c.get("title") or "",
                "creator": c.get("creator_name") or "",
                "views": views,
                "mp4": mp4,
                "url": c.get("url") or "",
                "created_at": c.get("created_at") or "",
            })

    if not all_candidates:
        print("No hay clips candidatos con los filtros actuales.")
        return

    # random sorpresa
    random.shuffle(all_candidates)
    picked = all_candidates[:max(1, TWITCH_PICK_TOTAL)]

    print(f"Seleccionados {len(picked)} clips para subir a inbox.")

    for c in picked:
        mp4_url = c["mp4"]
        views = c.get("views", 0)
        title = c.get("title", "")

        print("Downloading:", mp4_url)
        data = download_bytes(mp4_url)

        # key único
        h = hashlib.sha1(data).hexdigest()[:10]
        key = f"{R2_INBOX_PREFIX}/{iso_now()}__twitch__{c.get('id','unknown')}__{safe_slug(title)}__{h}.mp4"

        print("Uploading to R2 inbox:", key)
        upload_bytes_to_r2(key, data, "video/mp4")
        print(f"OK: {title[:60]} | views: {views}")

def timedelta_days(days: int):
    from datetime import timedelta
    return timedelta(days=int(max(0, days)))
