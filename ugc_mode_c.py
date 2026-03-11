import os
import json
import random
import subprocess
import tempfile
from datetime import datetime

import boto3


# =========================
# ENV HELPERS
# =========================

def env_nonempty(name, default=None):
    v = os.getenv(name)
    if not v:
        return default
    v = v.strip()
    return v if v else default


def env_int(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except:
        return default


# =========================
# CONFIG
# =========================

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

INPUT_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox_v2") or "ugc/inbox_v2").strip().strip("/")
INPUT_MANUAL_PREFIX = (env_nonempty("UGC_INBOX_MANUAL_PREFIX", "ugc/inbox_manual") or "ugc/inbox_manual").strip().strip("/")
OUTPUT_PREFIX = (env_nonempty("UGC_CLIPS_PREFIX", "ugc/library/clips") or "ugc/library/clips").strip().strip("/")

STATE_KEY = env_nonempty("MODE_C_STATE_KEY", "ugc/state/mode_c_state.json")

CLIP_SECONDS = env_int("MODE_C_CLIP_SECONDS", 8)
MAX_INPUTS = env_int("MODE_C_MAX_INPUTS", 5)
MAX_CLIPS_PER_VIDEO = env_int("MODE_C_MAX_CLIPS_PER_VIDEO", 3)


# =========================
# R2 CLIENT
# =========================

def r2():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


# =========================
# STATE
# =========================

def load_state():
    try:
        obj = r2().get_object(
            Bucket=BUCKET_NAME,
            Key=STATE_KEY
        )
        state = json.loads(obj["Body"].read())
    except Exception:
        state = {}

    if not isinstance(state, dict):
        state = {}

    if "processed" not in state:
        state["processed"] = []

    if not isinstance(state["processed"], list):
        state["processed"] = []

    return state


def save_state(state):
    if not isinstance(state, dict):
        state = {}

    if "processed" not in state:
        state["processed"] = []

    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=STATE_KEY,
        Body=json.dumps(state).encode(),
        ContentType="application/json",
    )


# =========================
# LIST VIDEOS
# =========================

def list_videos_from_prefix(prefix):
    resp = r2().list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=f"{prefix}/"
    )

    videos = []

    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".mp4"):
            videos.append(key)

    return videos


def list_all_input_videos():
    auto_videos = list_videos_from_prefix(INPUT_PREFIX)
    manual_videos = list_videos_from_prefix(INPUT_MANUAL_PREFIX)

    all_videos = auto_videos + manual_videos
    all_videos.sort()

    return all_videos


# =========================
# DOWNLOAD
# =========================

def download(key, path):
    r2().download_file(
        BUCKET_NAME,
        key,
        path
    )


# =========================
# UPLOAD
# =========================

def upload(path, key):
    r2().upload_file(
        path,
        BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": "video/mp4"}
    )


# =========================
# VIDEO DURATION
# =========================

def get_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    try:
        return float(p.stdout.strip())
    except:
        return 0


# =========================
# CUT CLIP
# =========================

def cut_clip(src, start, seconds, dst):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(seconds),
        "-c", "copy",
        dst,
    ]

    subprocess.run(cmd, capture_output=True)


# =========================
# MAIN
# =========================

def run_mode_c():
    print("===== UGC MODE C START =====")
    print("INPUT_PREFIX:", INPUT_PREFIX)
    print("INPUT_MANUAL_PREFIX:", INPUT_MANUAL_PREFIX)
    print("OUTPUT_PREFIX:", OUTPUT_PREFIX)
    print("STATE_KEY:", STATE_KEY)

    state = load_state()
    processed = set(state["processed"])

    videos = list_all_input_videos()

    print("Videos totales encontrados:", len(videos))

    count = 0

    for key in videos:
        if count >= MAX_INPUTS:
            break

        if key in processed:
            continue

        print("Procesando:", key)

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "video.mp4")
            download(key, src)

            duration = get_duration(src)

            if duration < CLIP_SECONDS:
                print("Video demasiado corto:", key)
                processed.add(key)
                continue

            for i in range(MAX_CLIPS_PER_VIDEO):
                start = random.uniform(
                    0,
                    max(1, duration - CLIP_SECONDS - 1)
                )

                out = os.path.join(tmp, f"clip{i}.mp4")

                cut_clip(
                    src,
                    start,
                    CLIP_SECONDS,
                    out
                )

                clip_key = f"{OUTPUT_PREFIX}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{i}.mp4"

                upload(out, clip_key)

                print("Clip creado:", clip_key)

        processed.add(key)
        count += 1

    state["processed"] = list(processed)[-5000:]
    save_state(state)

    print("===== MODE C DONE =====")


if __name__ == "__main__":
    run_mode_c()
