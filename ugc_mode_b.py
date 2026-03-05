import os
import re
import json
import time
import uuid
import random
import hashlib
import tempfile
import subprocess
from typing import Optional, Dict, Any, List, Tuple

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

# R2 / S3
AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or "").rstrip("/")

# UGC paths in R2
UGC_INBOX_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")
UGC_OUTPUT_PREFIX = (env_nonempty("UGC_OUTPUT_PREFIX", "ugc/outputs/reels") or "ugc/outputs/reels").strip().strip("/")

# Reel params
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
REEL_SECONDS = env_int("REEL_SECONDS", 12)

# HUD overlays
HUD_DIR = env_nonempty("HUD_DIR", "assets") or "assets"
HUD_PREFIX = env_nonempty("HUD_PREFIX", "hud_") or "hud_"
HUD_PROBABILITY = env_float("HUD_PROBABILITY", 0.85)  # 0..1
HUD_OPACITY = env_float("HUD_OPACITY", 0.75)          # 0..1

# Music (optional)
MUSIC_SEARCH_DIR = env_nonempty("MUSIC_SEARCH_DIR", "assets") or "assets"
MUSIC_PROBABILITY = env_float("MUSIC_PROBABILITY", 0.65)  # 0..1
MUSIC_VOLUME = env_float("MUSIC_VOLUME", 0.35)

# IG publish (optional)
ENABLE_IG_PUBLISH = env_bool("ENABLE_IG_PUBLISH", True)
IG_USER_ID = env_nonempty("IG_USER_ID")
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")
GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

# Safety / controls
DRY_RUN = env_bool("DRY_RUN", False)
MAX_ITEMS_PER_RUN = env_int("UGC_MAX_ITEMS", 6)


# -------------------------
# R2 helpers
# -------------------------

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

def r2_list_keys(prefix: str, max_keys: int = 50) -> List[str]:
    s3 = r2_client()
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": BUCKET_NAME, "Prefix": prefix, "MaxKeys": max_keys}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in (resp.get("Contents") or []):
            k = obj.get("Key")
            if k and not k.endswith("/"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def r2_download_to_file(key: str, dst_path: str) -> None:
    s3 = r2_client()
    s3.download_file(BUCKET_NAME, key, dst_path)

def r2_upload_file_public(local_path: str, key: str, content_type: str = "video/mp4") -> str:
    if not R2_PUBLIC_BASE_URL.startswith("http"):
        raise RuntimeError("R2_PUBLIC_BASE_URL inválido. Debe empezar por https://")

    s3 = r2_client()
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=f.read(), ContentType=content_type)

    return f"{R2_PUBLIC_BASE_URL}/{key}"

def r2_move_object(src_key: str, dst_key: str) -> None:
    s3 = r2_client()
    s3.copy_object(Bucket=BUCKET_NAME, CopySource={"Bucket": BUCKET_NAME, "Key": src_key}, Key=dst_key)
    s3.delete_object(Bucket=BUCKET_NAME, Key=src_key)


# -------------------------
# HUD + music picking
# -------------------------

def pick_hud_overlay() -> Optional[str]:
    if random.random() > max(0.0, min(1.0, HUD_PROBABILITY)):
        return None
    if not os.path.isdir(HUD_DIR):
        return None

    files = []
    for f in os.listdir(HUD_DIR):
        if f.startswith(HUD_PREFIX) and f.lower().endswith(".png"):
            files.append(os.path.join(HUD_DIR, f))

    if not files:
        return None
    return random.choice(files)

def list_mp3_files(search_dir: str) -> List[str]:
    out: List[str] = []
    if search_dir and os.path.isdir(search_dir):
        for root, _, fnames in os.walk(search_dir):
            for fn in fnames:
                if fn.lower().endswith(".mp3"):
                    out.append(os.path.join(root, fn))
    return out

def pick_music() -> Optional[str]:
    if random.random() > max(0.0, min(1.0, MUSIC_PROBABILITY)):
        return None
    candidates = list_mp3_files(MUSIC_SEARCH_DIR)
    # filtra mp3 muy pequeños
    good = []
    for c in candidates:
        try:
            if os.path.isfile(c) and os.path.getsize(c) > 50_000:
                good.append(c)
        except Exception:
            pass
    if not good:
        return None
    return random.choice(good)


# -------------------------
# FFmpeg reel builder
# -------------------------

def run_cmd(cmd: List[str], timeout: int = 480) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if p.returncode != 0:
        raise RuntimeError(
            "FFmpeg falló.\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDERR:\n{(p.stderr or '')[:4000]}"
        )

def build_reel(
    input_video: str,
    output_video: str,
    hud_png: Optional[str],
    music_mp3: Optional[str],
) -> None:
    """
    MODO 3: vertical 1080x1920 + HUD + glitch + zoom punch
    - zoom punch: micro-zooms (retención)
    - glitch: rgb shift + pequeños “saltos” sutiles
    - flash: micro-flashes
    """
    # Intensidades (ajústalas si quieres más/menos loco)
    ZOOM_BASE = 1.00
    ZOOM_PEAK = 1.08          # 1.05-1.12 recomendado
    GLITCH_STRENGTH = 0.006   # 0.003-0.012 recomendado
    FLASH_STRENGTH = 0.18     # 0.10-0.30 recomendado

    vf_parts = []

    # 1) Base: escala/crop a vertical y fps
    vf_parts.append(
        f"[0:v]scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
        f"crop={REEL_W}:{REEL_H},fps=30,format=rgba[v0];"
    )

    # 2) Zoom punch (sutil): zoom in/out con leve pan
    # Usamos zoompan para dar micro-movimiento. d=1 porque procesamos frame a frame a 30fps.
    # z sube y baja con sin() (punch). x/y mueven suavemente para evitar “static video”.
    vf_parts.append(
        f"[v0]zoompan="
        f"z='{ZOOM_BASE}+({ZOOM_PEAK-ZOOM_BASE})*abs(sin(2*PI*on/45))':"
        f"x='iw/2-(iw/zoom/2)+20*sin(2*PI*on/120)':"
        f"y='ih/2-(ih/zoom/2)+18*cos(2*PI*on/110)':"
        f"d=1:s={REEL_W}x{REEL_H}:fps=30,format=rgba[vz];"
    )

    # 3) Glitch (RGB split suave) + micro “jump”
    # - rgbashift hace separación RGB (glitch)
    # - rotate/translate muy sutil (para sensación de “impacto”)
    vf_parts.append(
        f"[vz]rgbashift=rh={GLITCH_STRENGTH}:rv=0:gh=0:gv={GLITCH_STRENGTH}:"
        f"bh={GLITCH_STRENGTH}:bv=0,format=rgba[vg];"
    )

    # 4) Flash punch (muy sutil) con eq: brillo sube y baja rápido
    vf_parts.append(
        f"[vg]eq=brightness='{FLASH_STRENGTH}*abs(sin(2*PI*t*0.9))':"
        f"contrast='1.0+0.10*abs(sin(2*PI*t*0.7))',format=rgba[vfx];"
    )

    # 5) HUD overlay (si existe) con opacidad
    if hud_png:
        vf_parts.append(
            f"[1:v]scale={REEL_W}:{REEL_H},format=rgba,"
            f"colorchannelmixer=aa={max(0.0, min(1.0, HUD_OPACITY))}[hud];"
        )
        vf_parts.append("[vfx][hud]overlay=0:0:format=auto[vout]")
        video_map = "[vout]"
    else:
        vf_parts.append("[vfx]copy[vout]")
        video_map = "[vout]"

    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error"]

    # inputs
    cmd += ["-i", input_video]
    if hud_png:
        cmd += ["-i", hud_png]
    if music_mp3:
        cmd += ["-i", music_mp3]

    cmd += ["-filter_complex", "".join(vf_parts)]
    cmd += ["-map", video_map]

    # audio
    if music_mp3:
        music_idx = 2 if hud_png else 1
        cmd += ["-map", f"{music_idx}:a", "-filter:a", f"volume={MUSIC_VOLUME}", "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]

    cmd += [
        "-t", str(int(REEL_SECONDS)),
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_video,
    ]

    run_cmd(cmd, timeout=600)


# -------------------------
# IG publish (optional)
# -------------------------

def ig_api_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.post(url, data=data, timeout=HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"IG POST failed: {r.status_code} {r.text[:1500]}")
    return r.json()

def ig_api_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"IG GET failed: {r.status_code} {r.text[:1500]}")
    return r.json()

def ig_wait_container(creation_id: str, access_token: str, timeout_sec: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        j = ig_api_get(f"{creation_id}", {"fields": "status_code", "access_token": access_token})
        st = (j.get("status_code") or "").upper()
        print("IG status:", st)
        if st in ("FINISHED", "PUBLISHED"):
            return
        if st in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {j}")
        time.sleep(3)
    raise TimeoutError("IG container timeout")

def ig_publish_reel(video_url: str, caption: str) -> Dict[str, Any]:
    if not (IG_USER_ID and IG_ACCESS_TOKEN):
        raise RuntimeError("Faltan IG_USER_ID o IG_ACCESS_TOKEN")

    print("Creating IG container")
    j = ig_api_post(
        f"{IG_USER_ID}/media",
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": IG_ACCESS_TOKEN,
        },
    )
    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG create failed: {j}")

    print("Waiting IG container:", creation_id)
    ig_wait_container(creation_id, IG_ACCESS_TOKEN)

    print("Publishing IG")
    res = ig_api_post(f"{IG_USER_ID}/media_publish", {"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN})
    return res


# -------------------------
# Caption (simple)
# -------------------------

def build_caption_for_clip(src_key: str) -> str:
    # caption sencillo (puedes cambiarlo por OpenAI si quieres)
    base = os.path.basename(src_key)
    return (
        "🎮 Momento gamer del día… y pasó de verdad 😳🔥\n"
        "¿Te ha pasado algo así jugando? 👇\n\n"
        "#gaming #esports #gamer #clips #reels"
        f"\n\n({base})"
    )


# -------------------------
# Main runner
# -------------------------

def run_mode_b() -> None:
    print("UGC MODE B START")
    print("ENV:")
    print(" - DRY_RUN:", DRY_RUN)
    print(" - R2_PUBLIC_BASE_URL set:", bool(R2_PUBLIC_BASE_URL))
    print(" - UGC_INBOX_PREFIX:", UGC_INBOX_PREFIX)
    print(" - UGC_OUTPUT_PREFIX:", UGC_OUTPUT_PREFIX)
    print(" - HUD_DIR:", HUD_DIR)
    print(" - ENABLE_IG_PUBLISH:", ENABLE_IG_PUBLISH)

    inbox_prefix = f"{UGC_INBOX_PREFIX}/"
    keys = r2_list_keys(inbox_prefix, max_keys=200)

    # solo mp4
    keys = [k for k in keys if k.lower().endswith(".mp4")]
    keys.sort()

    if not keys:
        print("Inbox vacío. Nada que procesar ✅")
        return

    processed = 0

    for key in keys:
        if processed >= MAX_ITEMS_PER_RUN:
            break

        print("Processing:", key)

        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.mp4")
            out_path = os.path.join(td, "reel.mp4")

            # download
            r2_download_to_file(key, in_path)

            # pick HUD + music
            hud = pick_hud_overlay()
            music = pick_music()

            print("HUD:", hud if hud else "NONE")
            print("MUSIC:", music if music else "NONE")

            # build reel
            build_reel(
                input_video=in_path,
                output_video=out_path,
                hud_png=hud,
                music_mp3=music,
            )

            # upload output
            with open(out_path, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()[:10]

            out_key = f"{UGC_OUTPUT_PREFIX}/{h}.mp4"
            out_url = r2_upload_file_public(out_path, out_key)

            print("Uploaded reel:", out_url)

            # publish IG
            caption = build_caption_for_clip(key)
            if DRY_RUN:
                print("[DRY_RUN] Caption:\n", caption)
            else:
                if ENABLE_IG_PUBLISH and IG_USER_ID and IG_ACCESS_TOKEN:
                    res = ig_publish_reel(out_url, caption)
                    print("IG publish:", res)
                else:
                    print("IG publish skipped (disabled o faltan tokens).")

            # move processed inbox item to archive (para no repetir)
            archive_key = key.replace(f"{UGC_INBOX_PREFIX}/", f"{UGC_INBOX_PREFIX}_done/", 1)
            if DRY_RUN:
                print("[DRY_RUN] No muevo archivo en inbox.")
            else:
                r2_move_object(key, archive_key)
                print("Moved inbox item to:", archive_key)

        processed += 1

    print("UGC MODE B DONE")
