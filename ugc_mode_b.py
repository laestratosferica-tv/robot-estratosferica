import os
import time
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


# -------------------------
# GLOBAL CONFIG
# -------------------------

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or "").rstrip("/")

UGC_INBOX_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")
UGC_OUTPUT_PREFIX = (env_nonempty("UGC_OUTPUT_PREFIX", "ugc/outputs/reels") or "ugc/outputs/reels").strip().strip("/")

REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
REEL_SECONDS = env_int("REEL_SECONDS", 12)

HUD_DIR = env_nonempty("HUD_DIR", "assets") or "assets"
HUD_PREFIX = env_nonempty("HUD_PREFIX", "hud_") or "hud_"
HUD_PROBABILITY = env_float("HUD_PROBABILITY", 0.85)

MUSIC_SEARCH_DIR = env_nonempty("MUSIC_SEARCH_DIR", "assets") or "assets"
MUSIC_PROBABILITY = env_float("MUSIC_PROBABILITY", 0.65)

GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", True)
IG_USER_ID = env_nonempty("IG_USER_ID")
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")

ENABLE_FB_PUBLISH = env_bool("ENABLE_FB_PUBLISH", False)
FB_PAGE_ID = env_nonempty("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = env_nonempty("FB_PAGE_ACCESS_TOKEN")

DRY_RUN = env_bool("DRY_RUN", False)
MAX_ITEMS_PER_RUN = env_int("UGC_MAX_ITEMS", 6)


# -------------------------
# R2 helpers
# -------------------------

def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def r2_list_keys(prefix: str) -> List[str]:
    s3 = r2_client()
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    keys = []
    for obj in resp.get("Contents", []):
        k = obj["Key"]
        if not k.endswith("/"):
            keys.append(k)
    return keys


def r2_download_to_file(key: str, dst_path: str):
    s3 = r2_client()
    s3.download_file(BUCKET_NAME, key, dst_path)


def r2_upload_file_public(local_path: str, key: str):
    s3 = r2_client()
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=f.read(), ContentType="video/mp4")
    return f"{R2_PUBLIC_BASE_URL}/{key}"


def r2_move_object(src_key: str, dst_key: str):
    s3 = r2_client()
    s3.copy_object(Bucket=BUCKET_NAME, CopySource={"Bucket": BUCKET_NAME, "Key": src_key}, Key=dst_key)
    s3.delete_object(Bucket=BUCKET_NAME, Key=src_key)


# -------------------------
# HUD / music
# -------------------------

def pick_hud_overlay():
    if random.random() > HUD_PROBABILITY:
        return None
    if not os.path.isdir(HUD_DIR):
        return None

    files = []
    for f in os.listdir(HUD_DIR):
        if f.startswith(HUD_PREFIX) and f.endswith(".png"):
            files.append(os.path.join(HUD_DIR, f))

    if not files:
        return None

    return random.choice(files)


def list_mp3_files(search_dir):
    out = []
    if os.path.isdir(search_dir):
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".mp3"):
                    out.append(os.path.join(root, f))
    return out


def pick_music():
    if random.random() > MUSIC_PROBABILITY:
        return None

    candidates = list_mp3_files(MUSIC_SEARCH_DIR)
    if not candidates:
        return None

    return random.choice(candidates)


# -------------------------
# FFmpeg builder
# -------------------------

def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)


def build_reel(input_video, output_video, hud_png, music_mp3):
    vf = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[v0];"

    if hud_png:
        vf += "[1:v]scale=1080:1920[h];[v0][h]overlay=0:0[v1];"
        vout = "[v1]"
    else:
        vf += "[v0]copy[v1];"
        vout = "[v1]"

    cmd = ["ffmpeg", "-y", "-i", input_video]

    if hud_png:
        cmd += ["-i", hud_png]

    if music_mp3:
        cmd += ["-i", music_mp3]

    cmd += ["-filter_complex", vf, "-map", vout]

    if music_mp3:
        idx = 2 if hud_png else 1
        cmd += ["-map", f"{idx}:a"]
    else:
        cmd += ["-an"]

    cmd += [
        "-t",
        str(REEL_SECONDS),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_video,
    ]

    run_cmd(cmd)


# -------------------------
# IG publish
# -------------------------

def ig_publish_reel(video_url, caption):
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

    if r.status_code >= 400:
        raise RuntimeError(f"IG create failed: {r.status_code} {r.text}")

    j = r.json()
    creation_id = j["id"]

    while True:
        s = requests.get(
            f"{GRAPH_BASE}/{creation_id}",
            params={"fields": "status_code", "access_token": IG_ACCESS_TOKEN},
            timeout=HTTP_TIMEOUT,
        )

        if s.status_code >= 400:
            raise RuntimeError(f"IG status failed: {s.status_code} {s.text}")

        sj = s.json()
        print("IG status:", sj.get("status_code"))

        if sj.get("status_code") == "FINISHED":
            break

        if sj.get("status_code") in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {sj}")

        time.sleep(3)

    r = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media_publish",
        data={"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN},
        timeout=HTTP_TIMEOUT,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"IG publish failed: {r.status_code} {r.text}")

    return r.json()


# -------------------------
# FB upload (hosted reel via file_url)
# -------------------------

def fb_start_reel_upload(file_size):
    r = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "START",
            "file_size": str(file_size),
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"FB START failed: {r.status_code} {r.text}")

    return r.json()


def fb_transfer_reel_hosted(upload_url, public_video_url):
    headers = {
        "Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}",
        "file_url": public_video_url,
    }

    r = requests.post(
        upload_url,
        headers=headers,
        timeout=HTTP_TIMEOUT,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"FB TRANSFER failed: {r.status_code} {r.text}")

    return r.json()


def fb_finish_reel_upload(video_id, caption):
    r = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "FINISH",
            "video_id": video_id,
            "video_state": "PUBLISHED",
            "description": caption,
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"FB FINISH failed: {r.status_code} {r.text}")

    return r.json()


def fb_publish_reel(public_video_url, local_video_path, caption):
    print("FB reel upload START")

    size = os.path.getsize(local_video_path)
    start = fb_start_reel_upload(size)

    upload_url = start.get("upload_url")
    video_id = start.get("video_id")

    if not upload_url or not video_id:
        raise RuntimeError(f"FB START inválido: {start}")

    print("FB reel upload TRANSFER")
    transfer = fb_transfer_reel_hosted(upload_url, public_video_url)
    print("FB transfer:", transfer)

    print("FB reel upload FINISH")
    finish = fb_finish_reel_upload(video_id, caption)

    return finish


# -------------------------
# caption
# -------------------------

def build_caption_for_clip(src_key):
    base = os.path.basename(src_key)

    return (
        "🎮 Momento gamer del día 😳🔥\n"
        "¿Te ha pasado algo así jugando?\n\n"
        "#gaming #esports #clips\n\n"
        f"({base})"
    )


# -------------------------
# main
# -------------------------

def run_mode_b():
    print("UGC MODE B START")
    print("ENV:")
    print(" - DRY_RUN:", DRY_RUN)
    print(" - R2_PUBLIC_BASE_URL set:", bool(R2_PUBLIC_BASE_URL))
    print(" - UGC_INBOX_PREFIX:", UGC_INBOX_PREFIX)
    print(" - UGC_OUTPUT_PREFIX:", UGC_OUTPUT_PREFIX)
    print(" - HUD_DIR:", HUD_DIR)
    print(" - ENABLE_IG_PUBLISH:", ENABLE_IG_PUBLISH)
    print(" - ENABLE_FB_PUBLISH:", ENABLE_FB_PUBLISH)

    inbox_prefix = f"{UGC_INBOX_PREFIX}/"
    keys = r2_list_keys(inbox_prefix)
    keys = [k for k in keys if k.endswith(".mp4")]
    keys.sort()

    if not keys:
        print("Inbox vacío")
        return

    processed = 0

    for key in keys:
        if processed >= MAX_ITEMS_PER_RUN:
            break

        print("Processing:", key)

        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.mp4")
            out_path = os.path.join(td, "reel.mp4")

            r2_download_to_file(key, in_path)

            hud = pick_hud_overlay()
            music = pick_music()

            print("HUD:", hud if hud else "NONE")
            print("MUSIC:", music if music else "NONE")

            build_reel(in_path, out_path, hud, music)

            with open(out_path, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()[:10]

            out_key = f"{UGC_OUTPUT_PREFIX}/{h}.mp4"
            out_url = r2_upload_file_public(out_path, out_key)

            print("Uploaded reel:", out_url)

            caption = build_caption_for_clip(key)

            if DRY_RUN:
                print("[DRY_RUN] No publica ni mueve archivo.")
            else:
                if ENABLE_IG_PUBLISH and IG_USER_ID and IG_ACCESS_TOKEN:
                    ig = ig_publish_reel(out_url, caption)
                    print("IG publish:", ig)
                else:
                    print("IG publish skipped (disabled o faltan tokens).")

                if ENABLE_FB_PUBLISH and FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN:
                    fb = fb_publish_reel(out_url, out_path, caption)
                    print("FB publish:", fb)
                else:
                    print("FB publish skipped (disabled o faltan tokens).")

                archive_key = key.replace(
                    f"{UGC_INBOX_PREFIX}/",
                    f"{UGC_INBOX_PREFIX}_done/",
                    1,
                )

                r2_move_object(key, archive_key)
                print("Moved inbox item to:", archive_key)

        processed += 1

    print("UGC MODE B DONE")
