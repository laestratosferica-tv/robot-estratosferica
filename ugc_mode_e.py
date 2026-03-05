# ugc_mode_e.py
# AUTO EXPLORER (TWITCH) -> R2 INBOX

import os
import time
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests
import boto3

# =========================
# ENV (Twitch)
# =========================
TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

MAX_CLIPS = int(os.getenv("TWITCH_MAX_CLIPS_PER_RUN", "3"))
LOOKBACK_HOURS = int(os.getenv("TWITCH_LOOKBACK_HOURS", "24"))
MIN_VIEWS = int(os.getenv("TWITCH_MIN_VIEWS", "3000"))

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

# =========================
# ENV (R2)
# =========================
BUCKET_NAME = os.getenv("BUCKET_NAME")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

UGC_INBOX_PREFIX = os.getenv("UGC_INBOX_PREFIX", "ugc/inbox/").strip()
if not UGC_INBOX_PREFIX.endswith("/"):
    UGC_INBOX_PREFIX += "/"


# =========================
# R2 CLIENT
# =========================
def r2_client():
    if not (BUCKET_NAME and R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise RuntimeError("Faltan env R2: BUCKET_NAME, R2_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def s3_put_bytes(key: str, data: bytes, content_type: str):
    r2_client().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


# =========================
# Twitch Auth
# =========================
def get_app_token() -> str:
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
    return r.json()["access_token"]


def headers(token: str) -> Dict[str, str]:
    return {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {token}",
        "User-Agent": "ugc-trend-bot",
    }


# =========================
# Twitch Data
# =========================
def get_top_games(token: str) -> List[Dict]:
    url = "https://api.twitch.tv/helix/games/top"
    r = requests.get(url, headers=headers(token), params={"first": 20}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json().get("data", [])


def get_clips(token: str, game_id: str) -> List[Dict]:
    start = datetime.utcnow() - timedelta(hours=LOOKBACK_HOURS)
    params = {
        "game_id": game_id,
        "started_at": start.isoformat("T") + "Z",
        "first": 20,
    }
    r = requests.get("https://api.twitch.tv/helix/clips", headers=headers(token), params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json().get("data", [])


def score_clip(clip: Dict) -> float:
    views = int(clip.get("view_count", 0))
    created_at = clip.get("created_at", "")
    try:
        age = datetime.utcnow() - datetime.fromisoformat(created_at.replace("Z", "+00:00")).replace(tzinfo=None)
        age_hours = age.total_seconds() / 3600
    except Exception:
        age_hours = 9999

    # score simple: views + bonus por frescura
    score = (views * 0.7) + (1000 / (1 + age_hours))
    return float(score)


# =========================
# Download clip bytes
# =========================
def twitch_mp4_url_from_thumbnail(thumbnail_url: str) -> str:
    # Twitch: thumbnail_url contiene "-preview-..." -> mp4 es antes de "-preview" + ".mp4"
    return thumbnail_url.split("-preview")[0] + ".mp4"


def download_clip_bytes(mp4_url: str) -> bytes:
    r = requests.get(mp4_url, timeout=60, headers={"User-Agent": "ugc-trend-bot"})
    r.raise_for_status()
    return r.content


# =========================
# Upload to R2 inbox
# =========================
def make_inbox_key(clip: Dict) -> str:
    # para ordenar y evitar colisiones: fecha + game + id
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cid = clip.get("id", "unknown")
    game_id = clip.get("game_id", "game")
    return f"{UGC_INBOX_PREFIX}{today}__twitch__{game_id}__{cid}.mp4"


# =========================
# Main
# =========================
def run_mode_e():
    print("===== MODE E (TWITCH) -> R2 INBOX =====")

    token = get_app_token()
    games = get_top_games(token)

    candidates: List[Dict] = []

    for g in games[:10]:
        name = g.get("name", "")
        gid = g.get("id", "")
        if not gid:
            continue

        print("Exploring:", name)
        clips = get_clips(token, gid)

        for c in clips:
            views = int(c.get("view_count", 0))
            if views < MIN_VIEWS:
                continue
            c["score"] = score_clip(c)
            candidates.append(c)

    candidates.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    picks = candidates[:MAX_CLIPS]
    if not picks:
        print("No hay clips que cumplan MIN_VIEWS.")
        return

    print(f"Seleccionados {len(picks)} clips para subir a inbox.")

    for clip in picks:
        title = clip.get("title", "")
        views = clip.get("view_count", 0)
        thumb = clip.get("thumbnail_url", "")
        if not thumb:
            continue

        mp4_url = twitch_mp4_url_from_thumbnail(thumb)
        print("Downloading:", mp4_url)
        video_bytes = download_clip_bytes(mp4_url)

        key = make_inbox_key(clip)
        print("Uploading to R2 inbox:", key)
        s3_put_bytes(key, video_bytes, "video/mp4")

        print("OK:", title, "| views:", views)

        time.sleep(1.0)

    print("===== MODE E DONE =====")


if __name__ == "__main__":
    run_mode_e()
