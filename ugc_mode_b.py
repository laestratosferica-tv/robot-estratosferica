import os
import re
import time
import json
import random
import hashlib
import tempfile
import subprocess
from typing import Optional, List, Dict, Any

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
HUD_PROBABILITY = env_float("HUD_PROBABILITY", 0.25)
HUD_OPACITY = env_float("HUD_OPACITY", 0.18)

MUSIC_SEARCH_DIR = env_nonempty("MUSIC_SEARCH_DIR", "assets") or "assets"
MUSIC_PROBABILITY = env_float("MUSIC_PROBABILITY", 0.20)
MUSIC_VOLUME = env_float("MUSIC_VOLUME", 0.30)

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

SAVE_DEBUG_REEL = env_bool("SAVE_DEBUG_REEL", True)
DEBUG_REEL_NAME = env_nonempty("DEBUG_REEL_NAME", "debug_last_reel.mp4") or "debug_last_reel.mp4"
DEBUG_INPUT_NAME = env_nonempty("DEBUG_INPUT_NAME", "debug_input.mp4") or "debug_input.mp4"

MAX_START_OFFSET_SECONDS = env_float("MAX_START_OFFSET_SECONDS", 8.0)

USER_AGENT = env_nonempty(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
) or "Mozilla/5.0"


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
    keys: List[str] = []
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": BUCKET_NAME,
            "Prefix": prefix,
            "MaxKeys": 200,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if not k.endswith("/"):
                keys.append(k)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    return keys


def r2_download_to_file(key: str, dst_path: str):
    r2_client().download_file(BUCKET_NAME, key, dst_path)


def r2_download_json(key: str) -> Dict[str, Any]:
    s3 = r2_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    raw = obj["Body"].read().decode("utf-8", errors="replace")
    return json.loads(raw)


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
    if random.random() > max(0.0, min(1.0, HUD_PROBABILITY)):
        return None

    if not os.path.isdir(HUD_DIR):
        return None

    files = []

    for f in os.listdir(HUD_DIR):
        name = f.lower()

        if not name.startswith(HUD_PREFIX.lower()):
            continue
        if not name.endswith(".png"):
            continue

        if (
            "safearea" in name
            or "guide" in name
            or "guides" in name
            or "template" in name
            or "layout" in name
            or "grid" in name
        ):
            continue

        path = os.path.join(HUD_DIR, f)

        try:
            if os.path.getsize(path) < 5000:
                continue
        except Exception:
            continue

        files.append(path)

    if not files:
        return None

    return random.choice(files)


def list_mp3_files(search_dir):
    out = []
    if os.path.isdir(search_dir):
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.lower().endswith(".mp3"):
                    out.append(os.path.join(root, f))
    return out


def pick_music():
    if random.random() > max(0.0, min(1.0, MUSIC_PROBABILITY)):
        return None

    candidates = list_mp3_files(MUSIC_SEARCH_DIR)
    if not candidates:
        return None

    good = []
    for c in candidates:
        try:
            if os.path.getsize(c) > 50_000:
                good.append(c)
        except Exception:
            pass

    if not good:
        return None

    return random.choice(good)


# -------------------------
# General helpers
# -------------------------

def copy_file(src: str, dst: str):
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())


def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDERR:\n{p.stderr}"
        )


def ffprobe_json(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    try:
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def get_video_duration_seconds(path: str) -> float:
    info = ffprobe_json(path)
    try:
        return float(info.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        return 0.0


def choose_start_offset(input_video: str) -> float:
    duration = get_video_duration_seconds(input_video)
    usable = duration - float(REEL_SECONDS)

    if usable <= 0:
        return 0.0

    max_offset = min(float(MAX_START_OFFSET_SECONDS), usable)

    if max_offset <= 1.0:
        return 0.0

    return round(random.uniform(1.5, max_offset), 3)


# -------------------------
# Resolve input video
# -------------------------

def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def download_http_file(url: str, dst_path: str, timeout: float = HTTP_TIMEOUT):
    s = requests_session()
    with s.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"Download failed {r.status_code}: {url}")

        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def extract_direct_video_url_from_twitch_page(page_url: str) -> Optional[str]:
    s = requests_session()
    r = s.get(page_url, timeout=HTTP_TIMEOUT, allow_redirects=True)

    if r.status_code >= 400:
        raise RuntimeError(f"Twitch page fetch failed: {r.status_code} {page_url}")

    html = r.text or ""

    patterns = [
        r'https://[^"\']+clips-media-assets[^"\']+\.mp4[^"\']*',
        r'https://[^"\']+cloudfront\.net/[^"\']+\.mp4[^"\']*',
        r'https://[^"\']+\.mp4[^"\']*',
    ]

    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return m.group(0).replace("\\u002F", "/").replace("\\/", "/")

    og_video = re.search(r'<meta[^>]+property=["\']og:video["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
    if og_video:
        candidate = og_video.group(1).replace("\\u002F", "/").replace("\\/", "/")
        if ".mp4" in candidate:
            return candidate

    twitter_player_stream = re.search(r'"videoQualities"\s*:\s*\[(.*?)\]', html, re.S)
    if twitter_player_stream:
        block = twitter_player_stream.group(1)
        m2 = re.search(r'"sourceURL"\s*:\s*"([^"]+)"', block)
        if m2:
            candidate = m2.group(1).replace("\\u002F", "/").replace("\\/", "/")
            if ".mp4" in candidate:
                return candidate

    return None


def resolve_json_to_video_url(meta: Dict[str, Any]) -> Optional[str]:
    for key in ("video_url", "mp4_url", "download_url", "direct_url"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    page_url = meta.get("url")
    if isinstance(page_url, str) and page_url.strip():
        return extract_direct_video_url_from_twitch_page(page_url.strip())

    return None


def prepare_input_video_from_key(src_key: str, dst_video_path: str) -> Dict[str, Any]:
    """
    Devuelve dict con info útil:
    {
      "src_type": "mp4" | "json",
      "meta": {...} | None,
      "resolved_video_url": "...optional..."
    }
    """
    lower = src_key.lower()

    if lower.endswith(".mp4"):
        r2_download_to_file(src_key, dst_video_path)
        return {
            "src_type": "mp4",
            "meta": None,
            "resolved_video_url": None,
        }

    if lower.endswith(".json"):
        meta = r2_download_json(src_key)
        video_url = resolve_json_to_video_url(meta)

        if not video_url:
            raise RuntimeError(
                f"No pude resolver video_url desde JSON: {src_key}. "
                "Idealmente el JSON debe traer video_url o mp4_url."
            )

        download_http_file(video_url, dst_video_path)

        return {
            "src_type": "json",
            "meta": meta,
            "resolved_video_url": video_url,
        }

    raise RuntimeError(f"Formato no soportado en inbox: {src_key}")


# -------------------------
# FFmpeg builder
# -------------------------

def build_reel(input_video, output_video, hud_png, music_mp3):
    start_offset = choose_start_offset(input_video)
    print("VIDEO START OFFSET:", start_offset)

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_offset),
        "-i", input_video,
    ]

    vf_parts = []
    vf_parts.append(
        f"[0:v]"
        f"fps=30,"
        f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
        f"crop={REEL_W}:{REEL_H},"
        f"setsar=1,"
        f"format=rgba[v0];"
    )

    if hud_png:
        cmd += ["-i", hud_png]
        vf_parts.append(
            f"[1:v]"
            f"scale={REEL_W}:{REEL_H},"
            f"format=rgba,"
            f"colorchannelmixer=aa={max(0.0, min(1.0, HUD_OPACITY))}[hud];"
        )
        vf_parts.append("[v0][hud]overlay=0:0:format=auto,format=yuv420p[vout]")
        video_map = "[vout]"
    else:
        vf_parts.append("[v0]format=yuv420p[vout]")
        video_map = "[vout]"

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

def build_caption_for_clip(src_key, meta: Optional[Dict[str, Any]] = None):
    if meta:
        title = str(meta.get("title") or "").strip()
        if title:
            return (
                "🎮 Momento gamer del día 😳🔥\n"
                f"{title}\n\n"
                "¿Te ha pasado algo así jugando?\n\n"
                "#gaming #esports #clips"
            )

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
    print(" - ENABLE_IG_PUBLISH:", ENABLE_IG_PUBLISH)
    print(" - ENABLE_FB_PUBLISH:", ENABLE_FB_PUBLISH)

    inbox_prefix = f"{UGC_INBOX_PREFIX}/"
    keys = r2_list_keys(inbox_prefix)

    # ahora acepta mp4 y json
    keys = [k for k in keys if k.lower().endswith(".mp4") or k.lower().endswith(".json")]
    keys.sort()

    print("INBOX ITEMS FOUND:", len(keys))
    for preview_key in keys[:10]:
        print(" -", preview_key)

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

            info = prepare_input_video_from_key(key, in_path)
            meta = info.get("meta")
            resolved_video_url = info.get("resolved_video_url")

            print("SRC TYPE:", info.get("src_type"))
            if resolved_video_url:
                print("RESOLVED VIDEO URL:", resolved_video_url)

            if SAVE_DEBUG_REEL:
                debug_input_path = os.path.join(os.getcwd(), DEBUG_INPUT_NAME)
                copy_file(in_path, debug_input_path)
                print("Saved debug input:", debug_input_path)

            hud = pick_hud_overlay()
            music = pick_music()

            print("HUD:", hud if hud else "NONE")
            print("MUSIC:", music if music else "NONE")

            build_reel(in_path, out_path, hud, music)

            if SAVE_DEBUG_REEL:
                debug_path = os.path.join(os.getcwd(), DEBUG_REEL_NAME)
                copy_file(out_path, debug_path)
                print("Saved debug reel:", debug_path)

            with open(out_path, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()[:10]

            out_key = f"{UGC_OUTPUT_PREFIX}/{h}.mp4"
            out_url = r2_upload_file_public(out_path, out_key)

            print("Uploaded reel:", out_url)
            print("Uploaded reel size bytes:", os.path.getsize(out_path))

            caption = build_caption_for_clip(key, meta)

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


if __name__ == "__main__":
    run_mode_b()
