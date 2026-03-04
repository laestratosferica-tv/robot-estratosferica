# ugc_mode_b.py
# UGC MODE B ENGINE
# Inbox → Queue → AI Caption → Reel Render → Multi-platform Publish

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
    if not v:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
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
# ENV CORE
# -------------------------

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL") or "").rstrip("/")

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

# Instagram
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", True)
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
IG_USER_ID = env_nonempty("IG_USER_ID")

GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

UGC_DRY_RUN = env_bool("UGC_DRY_RUN", False)


# Facebook
ENABLE_FB_PUBLISH = env_bool("ENABLE_FB_PUBLISH", False)
FB_PAGE_ID = env_nonempty("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = env_nonempty("FB_PAGE_ACCESS_TOKEN")


# TikTok
ENABLE_TIKTOK_PUBLISH = env_bool("ENABLE_TIKTOK_PUBLISH", False)
TIKTOK_ACCESS_TOKEN = env_nonempty("TIKTOK_ACCESS_TOKEN")
TIKTOK_OPEN_ID = env_nonempty("TIKTOK_OPEN_ID")


# YouTube
ENABLE_YT_PUBLISH = env_bool("ENABLE_YT_PUBLISH", False)
YOUTUBE_CLIENT_ID = env_nonempty("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = env_nonempty("YOUTUBE_CLIENT_SECRET")
YOUTUBE_REFRESH_TOKEN = env_nonempty("YOUTUBE_REFRESH_TOKEN")


# OpenAI
OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TTS_MODEL = env_nonempty("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")


# -------------------------
# UGC CONFIG
# -------------------------

UGC_INBOX_PREFIX = "ugc/inbox/"
UGC_QUEUE_PENDING = "ugc/queue/pending/"
UGC_QUEUE_PUBLISHED = "ugc/queue/published/"
UGC_QUEUE_FAILED = "ugc/queue/failed/"

UGC_OUTPUT_REELS_PREFIX = "ugc/outputs/reels/"
UGC_LIBRARY_PREFIX = "ugc/library/raw/"
UGC_PROCESSED_PREFIX = "ugc/processed/"
UGC_FAILED_PREFIX = "ugc/failed/"

UGC_STATE_KEY = "ugc/state/state.json"

MAX_POSTS_PER_DAY = env_int("MAX_POSTS_PER_DAY", 1)

LOCAL_TZ = ZoneInfo("America/Bogota")

REEL_W = 1080
REEL_H = 1920

UGC_CAP_SECONDS = 30
UGC_MIN_SECONDS = 5
UGC_FFMPEG_TIMEOUT = 900


# -------------------------
# R2
# -------------------------

def r2_client():

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def s3_get_bytes(key: str) -> bytes:

    s3 = r2_client()

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)

    return obj["Body"].read()


def s3_put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream"):

    r2_client().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_put_json(key: str, payload: Dict[str, Any]):

    s3_put_bytes(
        key,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        "application/json",
    )


def s3_get_json(key: str):

    try:

        return json.loads(s3_get_bytes(key).decode("utf-8"))

    except:

        return None


def r2_public_url(key: str):

    return f"{R2_PUBLIC_BASE_URL}/{key}"


# -------------------------
# OPENAI
# -------------------------

def openai_text(prompt: str):

    headers = {

        "Authorization": f"Bearer {OPENAI_API_KEY}",

        "Content-Type": "application/json",

    }

    payload = {

        "model": OPENAI_MODEL,

        "input": prompt,

    }

    r = requests.post(

        "https://api.openai.com/v1/responses",

        headers=headers,

        json=payload,

        timeout=60,

    )

    r.raise_for_status()

    return r.json()["output"][0]["content"][0]["text"]


# -------------------------
# VIDEO
# -------------------------

def make_reel(in_mp4, out_mp4):

    cmd = [

        "ffmpeg",

        "-y",

        "-i",

        in_mp4,

        "-vf",

        f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,crop={REEL_W}:{REEL_H}",

        "-t",

        str(UGC_CAP_SECONDS),

        "-c:v",

        "libx264",

        "-pix_fmt",

        "yuv420p",

        out_mp4,

    ]

    subprocess.run(cmd, check=True)


# -------------------------
# SOCIAL
# -------------------------

def ig_publish(video_url, caption):

    if UGC_DRY_RUN:

        print("DRY RUN IG", video_url)

        return

    r = requests.post(

        f"{GRAPH_BASE}/{IG_USER_ID}/media",

        data={

            "media_type": "REELS",

            "video_url": video_url,

            "caption": caption,

            "access_token": IG_ACCESS_TOKEN,

        },

    )

    creation_id = r.json()["id"]

    time.sleep(5)

    requests.post(

        f"{GRAPH_BASE}/{IG_USER_ID}/media_publish",

        data={

            "creation_id": creation_id,

            "access_token": IG_ACCESS_TOKEN,

        },

    )


def fb_publish(video_url, caption):

    if not ENABLE_FB_PUBLISH:

        return

    requests.post(

        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",

        data={

            "video_url": video_url,

            "description": caption,

            "access_token": FB_PAGE_ACCESS_TOKEN,

        },

    )


def tiktok_publish(video_url, caption):

    if not ENABLE_TIKTOK_PUBLISH:

        return

    requests.post(

        "https://open.tiktokapis.com/v2/post/publish/video/init/",

        headers={"Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}"},

        json={

            "open_id": TIKTOK_OPEN_ID,

            "source_info": {

                "source": "PULL_FROM_URL",

                "video_url": video_url,

            },

            "post_info": {

                "title": caption[:100],

            },

        },

    )


def youtube_publish(video_path, caption):

    if not ENABLE_YT_PUBLISH:

        return

    from google.oauth2.credentials import Credentials

    from googleapiclient.discovery import build

    from googleapiclient.http import MediaFileUpload

    creds = Credentials(

        None,

        refresh_token=YOUTUBE_REFRESH_TOKEN,

        token_uri="https://oauth2.googleapis.com/token",

        client_id=YOUTUBE_CLIENT_ID,

        client_secret=YOUTUBE_CLIENT_SECRET,

        scopes=["https://www.googleapis.com/auth/youtube.upload"],

    )

    youtube = build("youtube", "v3", credentials=creds)

    media = MediaFileUpload(video_path, mimetype="video/mp4")

    youtube.videos().insert(

        part="snippet,status",

        body={

            "snippet": {

                "title": caption[:80],

                "description": caption,

            },

            "status": {"privacyStatus": "public"},

        },

        media_body=media,

    ).execute()


# -------------------------
# MAIN WORKER
# -------------------------

def run_mode_b():

    print("UGC MODE B START")

    keys = r2_client().list_objects_v2(

        Bucket=BUCKET_NAME,

        Prefix=UGC_INBOX_PREFIX,

    ).get("Contents", [])

    for obj in keys:

        key = obj["Key"]

        if not key.endswith(".mp4"):

            continue

        print("Processing:", key)

        video = s3_get_bytes(key)

        with tempfile.TemporaryDirectory() as td:

            in_mp4 = f"{td}/in.mp4"

            out_mp4 = f"{td}/reel.mp4"

            open(in_mp4, "wb").write(video)

            make_reel(in_mp4, out_mp4)

            reel_bytes = open(out_mp4, "rb").read()

            reel_key = f"{UGC_OUTPUT_REELS_PREFIX}{short_hash(key)}.mp4"

            s3_put_bytes(reel_key, reel_bytes, "video/mp4")

            video_url = r2_public_url(reel_key)

            caption = openai_text(

                f"Haz caption viral gaming con pregunta final para este clip: {key}"

            )

            ig_publish(video_url, caption)

            fb_publish(video_url, caption)

            tiktok_publish(video_url, caption)

            try:

                youtube_publish(out_mp4, caption)

            except:

                pass

        print("Publicado:", video_url)

    print("UGC MODE B DONE")
