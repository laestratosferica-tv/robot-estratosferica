# ugc_mode_b.py
# UGC MODE B ENGINE
# Inbox → Reel Render → AI Caption → Publish

import os
import json
import time
import hashlib
import tempfile
import subprocess
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any

import boto3
import requests


# --------------------------------------------------
# ENV HELPERS
# --------------------------------------------------

def env_nonempty(name, default=None):
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")


def env_int(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except:
        return default


# --------------------------------------------------
# ENV
# --------------------------------------------------

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = env_nonempty("R2_PUBLIC_BASE_URL")

OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")

IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
IG_USER_ID = env_nonempty("IG_USER_ID")

GRAPH_VERSION = env_nonempty("GRAPH_VERSION", "v19.0")
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"

DRY_RUN = env_bool("DRY_RUN", False)

LOCAL_TZ = ZoneInfo("America/Bogota")


# --------------------------------------------------
# R2
# --------------------------------------------------

def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def s3_get_bytes(key):
    obj = r2_client().get_object(Bucket=BUCKET_NAME, Key=key)
    return obj["Body"].read()


def s3_put_bytes(key, data, content_type="application/octet-stream"):
    r2_client().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def r2_public_url(key):
    return f"{R2_PUBLIC_BASE_URL}/{key}"


# --------------------------------------------------
# OPENAI
# --------------------------------------------------

def openai_text(prompt):

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

    j = r.json()

    try:
        return j["output"][0]["content"][0]["text"]
    except:
        return str(j)


# --------------------------------------------------
# VIDEO
# --------------------------------------------------

def make_reel(in_mp4, out_mp4):

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_mp4,
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "-t",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_mp4,
    ]

    subprocess.run(cmd, check=True)


# --------------------------------------------------
# INSTAGRAM HELPERS
# --------------------------------------------------

def ig_api_post(path, data):

    r = requests.post(f"{GRAPH_BASE}/{path}", data=data)

    try:
        j = r.json()
    except:
        j = {"raw": r.text}

    if r.status_code >= 400:
        raise RuntimeError(f"IG POST ERROR {j}")

    return j


def ig_api_get(path, params):

    r = requests.get(f"{GRAPH_BASE}/{path}", params=params)

    try:
        j = r.json()
    except:
        j = {"raw": r.text}

    if r.status_code >= 400:
        raise RuntimeError(f"IG GET ERROR {j}")

    return j


def ig_wait_container(container_id):

    print("Waiting IG container:", container_id)

    for _ in range(120):

        j = ig_api_get(
            container_id,
            {
                "fields": "status_code",
                "access_token": IG_ACCESS_TOKEN,
            },
        )

        status = j.get("status_code")

        print("IG status:", status)

        if status == "FINISHED":
            return

        if status == "ERROR":
            raise RuntimeError("IG container failed")

        time.sleep(5)

    raise RuntimeError("IG timeout")


def ig_publish(video_url, caption):

    if DRY_RUN:
        print("DRY RUN IG", video_url)
        return

    print("Creating IG container")

    create = ig_api_post(
        f"{IG_USER_ID}/media",
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": IG_ACCESS_TOKEN,
        },
    )

    print("IG create:", create)

    creation_id = create["id"]

    ig_wait_container(creation_id)

    print("Publishing IG")

    publish = ig_api_post(
        f"{IG_USER_ID}/media_publish",
        {
            "creation_id": creation_id,
            "access_token": IG_ACCESS_TOKEN,
        },
    )

    print("IG publish:", publish)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def short_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def run_mode_b():

    print("UGC MODE B START")

    s3 = r2_client()

    objs = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix="ugc/inbox/",
    ).get("Contents", [])

    for obj in objs:

        key = obj["Key"]

        if not key.endswith(".mp4"):
            continue

        print("Processing:", key)

        video_bytes = s3_get_bytes(key)

        with tempfile.TemporaryDirectory() as td:

            in_mp4 = f"{td}/in.mp4"
            out_mp4 = f"{td}/reel.mp4"

            open(in_mp4, "wb").write(video_bytes)

            make_reel(in_mp4, out_mp4)

            reel_bytes = open(out_mp4, "rb").read()

            reel_key = f"ugc/outputs/reels/{short_hash(key)}.mp4"

            s3_put_bytes(reel_key, reel_bytes, "video/mp4")

            video_url = r2_public_url(reel_key)

            caption = openai_text(
                f"Haz caption viral gaming con pregunta final para este clip: {key}"
            )

            print("CAPTION:", caption)

            ig_publish(video_url, caption)

            print("Publicado:", video_url)

    print("UGC MODE B DONE")
