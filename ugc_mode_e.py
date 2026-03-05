# ugc_mode_e.py
import os
import requests
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict

TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

MAX_CLIPS = int(os.getenv("TWITCH_MAX_CLIPS_PER_RUN", "3"))
LOOKBACK_HOURS = int(os.getenv("TWITCH_LOOKBACK_HOURS", "24"))
MIN_VIEWS = int(os.getenv("TWITCH_MIN_VIEWS", "3000"))

# -------------------------
# Auth
# -------------------------

def get_app_token():

    url = "https://id.twitch.tv/oauth2/token"

    params = {
        "client_id": TWITCH_CLIENT_ID,
        "client_secret": TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials"
    }

    r = requests.post(url, params=params)
    r.raise_for_status()

    return r.json()["access_token"]


def headers(token):

    return {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }

# -------------------------
# Top Games
# -------------------------

def get_top_games(token):

    url = "https://api.twitch.tv/helix/games/top"

    r = requests.get(url, headers=headers(token), params={"first": 20})

    r.raise_for_status()

    return r.json()["data"]

# -------------------------
# Clips
# -------------------------

def get_clips(token, game_id):

    start = datetime.utcnow() - timedelta(hours=LOOKBACK_HOURS)

    params = {
        "game_id": game_id,
        "started_at": start.isoformat("T") + "Z",
        "first": 20
    }

    r = requests.get(
        "https://api.twitch.tv/helix/clips",
        headers=headers(token),
        params=params
    )

    r.raise_for_status()

    return r.json()["data"]

# -------------------------
# Viral Score
# -------------------------

def score_clip(clip):

    views = clip["view_count"]

    age = datetime.utcnow() - datetime.fromisoformat(
        clip["created_at"].replace("Z", "+00:00")
    )

    age_hours = age.total_seconds() / 3600

    score = (views * 0.7) + (1000 / (1 + age_hours))

    return score

# -------------------------
# Download clip
# -------------------------

def download_clip(clip):

    thumb = clip["thumbnail_url"]

    mp4 = thumb.split("-preview")[0] + ".mp4"

    print("Downloading:", mp4)

    r = requests.get(mp4)

    filename = f"ugc/inbox/{clip['id']}.mp4"

    with open(filename, "wb") as f:
        f.write(r.content)

    return filename

# -------------------------
# Main
# -------------------------

def run_mode_e():

    print("===== MODO E AUTO EXPLORER =====")

    token = get_app_token()

    games = get_top_games(token)

    candidates: List[Dict] = []

    for g in games[:10]:

        print("Exploring:", g["name"])

        clips = get_clips(token, g["id"])

        for c in clips:

            if c["view_count"] < MIN_VIEWS:
                continue

            c["score"] = score_clip(c)

            candidates.append(c)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    picks = candidates[:MAX_CLIPS]

    for clip in picks:

        print(
            "Selected:",
            clip["title"],
            "| views:",
            clip["view_count"]
        )

        download_clip(clip)

    print("Modo E terminado")
