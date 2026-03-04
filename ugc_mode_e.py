# ugc_mode_e.py
# SOURCE ENGINE - TWITCH CLIP HARVESTER

import os
import requests
import hashlib
from datetime import datetime

from ugc_mode_b import (
    r2_client,
    s3_put_bytes
)

TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

BUCKET_NAME = os.getenv("BUCKET_NAME")

INBOX_PREFIX = "ugc/inbox/"

TOP_STREAMERS = [
    "shroud",
    "s1mple",
    "tarik",
    "ninja",
    "xqc",
]


def short_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# -----------------------------------
# TWITCH AUTH
# -----------------------------------

def get_twitch_token():

    url = "https://id.twitch.tv/oauth2/token"

    params = {
        "client_id": TWITCH_CLIENT_ID,
        "client_secret": TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials",
    }

    r = requests.post(url, params=params)

    return r.json()["access_token"]


# -----------------------------------
# GET CLIPS
# -----------------------------------

def get_clips(user, token):

    url = "https://api.twitch.tv/helix/clips"

    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }

    params = {
        "broadcaster_id": user,
        "first": 5
    }

    r = requests.get(url, headers=headers, params=params)

    return r.json().get("data", [])


# -----------------------------------
# DOWNLOAD CLIP
# -----------------------------------

def download_clip(url):

    r = requests.get(url)

    return r.content


# -----------------------------------
# FIND STREAMER IDS
# -----------------------------------

def get_user_id(username, token):

    url = "https://api.twitch.tv/helix/users"

    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }

    params = {
        "login": username
    }

    r = requests.get(url, headers=headers, params=params)

    data = r.json()["data"]

    if not data:
        return None

    return data[0]["id"]


# -----------------------------------
# MAIN
# -----------------------------------

def run_mode_e():

    print("===== MODE E TWITCH HARVESTER =====")

    s3 = r2_client()

    token = get_twitch_token()

    for streamer in TOP_STREAMERS:

        print("Buscando clips:", streamer)

        user_id = get_user_id(streamer, token)

        if not user_id:
            continue

        clips = get_clips(user_id, token)

        for clip in clips:

            video_url = clip["thumbnail_url"].split("-preview")[0] + ".mp4"

            print("Descargando clip:", video_url)

            try:

                video_bytes = download_clip(video_url)

                key = f"{INBOX_PREFIX}{short_hash(video_url)}.mp4"

                s3_put_bytes(key, video_bytes, "video/mp4")

                print("Guardado en inbox:", key)

            except Exception as e:

                print("Error clip:", str(e))

    print("===== MODE E DONE =====")
