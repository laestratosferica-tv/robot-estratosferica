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
    try:
        return int(os.getenv(name, default))
    except:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except:
        return default


# -------------------------
# CONFIG
# -------------------------

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = env_nonempty("R2_PUBLIC_BASE_URL")

UGC_INBOX_PREFIX = env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox")
UGC_OUTPUT_PREFIX = env_nonempty("UGC_OUTPUT_PREFIX", "ugc/outputs/reels")

REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
REEL_SECONDS = env_int("REEL_SECONDS", 12)

HUD_DIR = env_nonempty("HUD_DIR", "assets")
HUD_PREFIX = env_nonempty("HUD_PREFIX", "hud_")
HUD_PROBABILITY = env_float("HUD_PROBABILITY", 0.7)
HUD_OPACITY = env_float("HUD_OPACITY", 0.35)

MUSIC_SEARCH_DIR = env_nonempty("MUSIC_SEARCH_DIR", "assets")
MUSIC_PROBABILITY = env_float("MUSIC_PROBABILITY", 0.5)
MUSIC_VOLUME = env_float("MUSIC_VOLUME", 0.35)

GRAPH_VERSION = env_nonempty("GRAPH_VERSION", "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

IG_USER_ID = env_nonempty("IG_USER_ID")
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")

MAX_ITEMS_PER_RUN = env_int("UGC_MAX_ITEMS", 6)


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


def r2_list_keys(prefix):

    s3 = r2_client()

    resp = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=prefix
    )

    keys = []

    for obj in resp.get("Contents", []):
        k = obj["Key"]

        if not k.endswith("/"):
            keys.append(k)

    return keys


def r2_download_to_file(key, dst):
    r2_client().download_file(BUCKET_NAME, key, dst)


def r2_upload_file_public(local_path, key):

    with open(local_path, "rb") as f:

        r2_client().put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=f.read(),
            ContentType="video/mp4"
        )

    return f"{R2_PUBLIC_BASE_URL}/{key}"


def r2_move_object(src, dst):

    s3 = r2_client()

    s3.copy_object(
        Bucket=BUCKET_NAME,
        CopySource={"Bucket": BUCKET_NAME, "Key": src},
        Key=dst,
    )

    s3.delete_object(
        Bucket=BUCKET_NAME,
        Key=src
    )


# -------------------------
# HUD / MUSIC
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
# FFmpeg
# -------------------------

def run_cmd(cmd):

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(p.stderr)


def build_reel(input_video, output_video, hud_png, music_mp3):

    vf_parts = []

    vf_parts.append(
        f"[0:v]scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
        f"crop={REEL_W}:{REEL_H},fps=30,format=rgba[v0];"
    )

    if hud_png:

        vf_parts.append(
            f"[1:v]scale={REEL_W}:{REEL_H},format=rgba,"
            f"colorchannelmixer=aa={HUD_OPACITY}[hud];"
        )

        vf_parts.append("[v0][hud]overlay=0:0:format=auto,format=yuv420p[vout]")

        video_map = "[vout]"

    else:

        vf_parts.append("[v0]format=yuv420p[vout]")

        video_map = "[vout]"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    cmd += ["-i", input_video]

    if hud_png:
        cmd += ["-i", hud_png]

    if music_mp3:
        cmd += ["-i", music_mp3]

    cmd += ["-filter_complex", "".join(vf_parts)]

    cmd += ["-map", video_map]

    if music_mp3:

        idx = 2 if hud_png else 1

        cmd += [
            "-map", f"{idx}:a",
            "-filter:a", f"volume={MUSIC_VOLUME}",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
        ]

    else:

        cmd += ["-an"]

    cmd += [
        "-t", str(REEL_SECONDS),
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
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
        raise RuntimeError(r.text)

    creation_id = r.json()["id"]

    while True:

        s = requests.get(
            f"{GRAPH_BASE}/{creation_id}",
            params={
                "fields": "status_code",
                "access_token": IG_ACCESS_TOKEN
            },
            timeout=HTTP_TIMEOUT,
        )

        status = s.json().get("status_code")

        print("IG status:", status)

        if status == "FINISHED":
            break

        if status in ("ERROR", "FAILED"):
            raise RuntimeError(s.json())

        time.sleep(3)

    r = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media_publish",
        data={
            "creation_id": creation_id,
            "access_token": IG_ACCESS_TOKEN
        },
        timeout=HTTP_TIMEOUT,
    )

    return r.json()


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
# MAIN
# -------------------------

def run_mode_b():

    print("UGC MODE B START")

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

            # debug local
            debug_path = os.path.join(os.getcwd(), "debug_last_reel.mp4")

            with open(out_path, "rb") as src, open(debug_path, "wb") as dst:
                dst.write(src.read())

            print("Saved debug reel:", debug_path)

            with open(out_path, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()[:10]

            out_key = f"{UGC_OUTPUT_PREFIX}/{h}.mp4"

            out_url = r2_upload_file_public(out_path, out_key)

            print("Uploaded reel:", out_url)

            caption = build_caption_for_clip(key)

            ig = ig_publish_reel(out_url, caption)

            print("IG publish:", ig)

            archive_key = key.replace(
                f"{UGC_INBOX_PREFIX}/",
                f"{UGC_INBOX_PREFIX}_done/",
                1,
            )

            r2_move_object(key, archive_key)

            print("Moved inbox item to:", archive_key)

        processed += 1

    print("UGC MODE B DONE")
