import os
import time
import json
import random
import hashlib
import tempfile
import subprocess
from typing import Optional, List

import requests
import boto3


# -------------------------
# ENV helpers
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
        return int(v)
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


# -------------------------
# GLOBAL CONFIG
# -------------------------

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = (
    env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or ""
).rstrip("/")

GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

IG_USER_ID = env_nonempty("IG_USER_ID")
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")

FB_PAGE_ID = env_nonempty("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = env_nonempty("FB_PAGE_ACCESS_TOKEN")

OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")


# -------------------------
# R2 helpers
# -------------------------

def r2_client():
    if not (R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise RuntimeError("Faltan credenciales R2")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def s3_get_bytes(key: str) -> bytes:
    obj = r2_client().get_object(Bucket=BUCKET_NAME, Key=key)
    return obj["Body"].read()


def s3_put_bytes(key: str, data: bytes, content_type="application/octet-stream"):
    r2_client().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def r2_public_url(key: str) -> str:
    return f"{R2_PUBLIC_BASE_URL}/{key}"


# -------------------------
# OpenAI helper
# -------------------------

def openai_text(prompt: str) -> str:

    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en secrets.")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json={
            "model": OPENAI_MODEL,
            "input": prompt
        },
        timeout=60,
    )

    if r.status_code >= 400:

        r2 = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )

        r2.raise_for_status()
        j = r2.json()

        return (j["choices"][0]["message"]["content"] or "").strip()

    j = r.json()

    if j.get("output_text"):
        return j["output_text"].strip()

    texts = []

    for item in j.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))

    return "\n".join(texts).strip()


# -------------------------
# Instagram publish
# -------------------------

def ig_publish(video_url: str, caption: str):

    if not (IG_USER_ID and IG_ACCESS_TOKEN):
        raise RuntimeError("Faltan IG_USER_ID o IG_ACCESS_TOKEN")

    r = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": IG_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )

    r.raise_for_status()

    creation_id = r.json()["id"]

    while True:

        s = requests.get(
            f"{GRAPH_BASE}/{creation_id}",
            params={
                "fields": "status_code",
                "access_token": IG_ACCESS_TOKEN,
            },
            timeout=HTTP_TIMEOUT,
        )

        status = s.json().get("status_code")

        if status == "FINISHED":
            break

        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container error: {status}")

        time.sleep(3)

    r = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media_publish",
        data={
            "creation_id": creation_id,
            "access_token": IG_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )

    r.raise_for_status()

    return r.json()


# -------------------------
# Facebook publish
# -------------------------

def fb_publish(video_url: str, caption: str):

    if not (FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN):
        raise RuntimeError("Faltan FB tokens")

    start = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "start",
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )

    start.raise_for_status()

    data = start.json()

    upload_url = data["upload_url"]
    video_id = data["video_id"]

    transfer = requests.post(
        upload_url,
        headers={
            "Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}",
            "file_url": video_url,
        },
        timeout=HTTP_TIMEOUT,
    )

    transfer.raise_for_status()

    finish = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "finish",
            "video_id": video_id,
            "video_state": "PUBLISHED",
            "description": caption,
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )

    finish.raise_for_status()

    return finish.json()


# -------------------------
# TikTok disabled
# -------------------------

def tiktok_publish(video_url: str, caption: str):
    return {"ok": False, "reason": "tiktok_disabled"}


# -------------------------
# YouTube disabled
# -------------------------

def youtube_publish(video_path: str, caption: str):
    return {"ok": False, "reason": "youtube_disabled"}
