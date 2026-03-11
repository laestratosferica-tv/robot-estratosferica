import os
import json
import random
import re
import subprocess
import tempfile
from datetime import datetime

import boto3


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
    except Exception:
        return default


AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

INPUT_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")
INPUT_MANUAL_PREFIX = (env_nonempty("UGC_INBOX_MANUAL_PREFIX", "ugc/inbox_manual") or "ugc/inbox_manual").strip().strip("/")
OUTPUT_PREFIX = (env_nonempty("UGC_CLIPS_PREFIX", "ugc/library/clips") or "ugc/library/clips").strip().strip("/")

STATE_KEY = env_nonempty("MODE_C_STATE_KEY", "ugc/state/mode_c_state.json")

CLIP_SECONDS = env_int("MODE_C_CLIP_SECONDS", 8)
MAX_INPUTS = env_int("MODE_C_MAX_INPUTS", 5)
MAX_CLIPS_PER_VIDEO = env_int("MODE_C_MAX_CLIPS_PER_VIDEO", 3)


def r2():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


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

    if "processed" not in state or not isinstance(state["processed"], list):
        state["processed"] = []

    return state


def save_state(state):
    if not isinstance(state, dict):
        state = {}

    if "processed" not in state or not isinstance(state["processed"], list):
        state["processed"] = []

    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=STATE_KEY,
        Body=json.dumps(state).encode(),
        ContentType="application/json",
    )


def list_videos_from_prefix(prefix):
    s3 = r2()
    videos = []
    token = None

    while True:
        params = {
            "Bucket": BUCKET_NAME,
            "Prefix": f"{prefix}/",
            "MaxKeys": 200,
        }

        if token:
            params["ContinuationToken"] = token

        resp = s3.list_objects_v2(**params)

        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".mp4"):
                videos.append(
                    {
                        "key": key,
                        "last_modified": obj.get("LastModified"),
                    }
                )

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    videos.sort(key=lambda x: x["last_modified"] or 0, reverse=True)
    return [x["key"] for x in videos]


def list_all_input_videos():
    auto_videos = list_videos_from_prefix(INPUT_PREFIX)
    manual_videos = list_videos_from_prefix(INPUT_MANUAL_PREFIX)

    all_videos = auto_videos + manual_videos
    return all_videos


def download(key, path):
    r2().download_file(
        BUCKET_NAME,
        key,
        path
    )


def upload(path, key):
    r2().upload_file(
        path,
        BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": "video/mp4"}
    )


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
    except Exception:
        return 0.0


def cut_clip(src, start, seconds, dst):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(seconds),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        dst,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0


def detect_game_from_key(key):
    text = (key or "").lower().replace("_", " ").replace("-", " ")

    checks = [
        ("valorant", "valorant"),
        ("cs2", "cs2"),
        ("counter strike", "cs2"),
        ("counter-strike", "cs2"),
        ("league of legends", "leagueoflegends"),
        ("lol", "leagueoflegends"),
        ("fortnite", "fortnite"),
        ("warzone", "warzone"),
        ("apex legends", "apex"),
        ("apex", "apex"),
        ("minecraft", "minecraft"),
        ("ea sports fc", "easportsfc"),
        ("fc", "easportsfc"),
        ("f1", "f1"),
        ("gran turismo", "granturismo"),
    ]

    for needle, label in checks:
        if needle in text:
            return label

    return "generic"


def safe_slug(text, max_len=60):
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "clip"
    return text[:max_len]


def build_clip_key(source_key, clip_index):
    game_slug = detect_game_from_key(source_key)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{OUTPUT_PREFIX}/{stamp}__{game_slug}__{clip_index}.mp4"


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
    print("Videos procesados en state:", len(processed))

    count = 0

    for key in videos:
        if count >= MAX_INPUTS:
            break

        if key in processed:
            print("SKIP already processed:", key)
            continue

        print("Procesando:", key)

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "video.mp4")
            download(key, src)

            duration = get_duration(src)
            print("Duración:", duration)

            if duration < CLIP_SECONDS:
                print("Video demasiado corto:", key)
                processed.add(key)
                continue

            created_for_video = 0

            for i in range(MAX_CLIPS_PER_VIDEO):
                max_start = max(1, duration - CLIP_SECONDS - 1)
                start = random.uniform(0, max_start)

                out = os.path.join(tmp, f"clip{i}.mp4")

                ok = cut_clip(src, start, CLIP_SECONDS, out)
                if not ok:
                    print("ERROR creando clip:", key, "clip", i)
                    continue

                if not os.path.exists(out) or os.path.getsize(out) == 0:
                    print("ERROR clip vacío:", key, "clip", i)
                    continue

                clip_key = build_clip_key(key, i)

                upload(out, clip_key)

                print("GAME DETECTED:", detect_game_from_key(key))
                print("Clip creado:", clip_key)

                created_for_video += 1

            if created_for_video == 0:
                print("No se creó ningún clip válido para:", key)

        processed.add(key)
        count += 1

    state["processed"] = list(processed)[-5000:]
    save_state(state)

    print("===== MODE C DONE =====")


if __name__ == "__main__":
    run_mode_c()
