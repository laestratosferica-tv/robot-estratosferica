# ===== INICIO: ugc_mode_c.py =====

import os
import json
import random
import re
import subprocess
import tempfile
import hashlib
from datetime import datetime, timezone

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


def env_float(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

INPUT_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")
INPUT_MANUAL_PREFIX = (env_nonempty("UGC_INBOX_MANUAL_PREFIX", "ugc/inbox_manual") or "ugc/inbox_manual").strip().strip("/")

OUTPUT_PREFIX = (env_nonempty("UGC_CLIPS_PREFIX", "ugc/library/clips") or "ugc/library/clips").strip().strip("/")
OUTPUT_META_PREFIX = (env_nonempty("UGC_CLIPS_META_PREFIX", "ugc/meta/clips") or "ugc/meta/clips").strip().strip("/")

STATE_KEY = env_nonempty("MODE_C_STATE_KEY", "ugc/state/mode_c_state.json")

CLIP_SECONDS = env_int("MODE_C_CLIP_SECONDS", 8)
MAX_INPUTS = env_int("MODE_C_MAX_INPUTS", 5)
MAX_CLIPS_PER_VIDEO = env_int("MODE_C_MAX_CLIPS_PER_VIDEO", 3)

# distancia mínima entre inicios de clips del mismo video
MODE_C_MIN_GAP_SECONDS = env_float("MODE_C_MIN_GAP_SECONDS", 18.0)

# evita cortar los primeros/últimos segundos del video
MODE_C_START_PADDING_SECONDS = env_float("MODE_C_START_PADDING_SECONDS", 6.0)
MODE_C_END_PADDING_SECONDS = env_float("MODE_C_END_PADDING_SECONDS", 6.0)

# si el video es corto, reduce exigencia
MODE_C_SHORT_VIDEO_THRESHOLD = env_float("MODE_C_SHORT_VIDEO_THRESHOLD", 40.0)

# para naming/meta
MODE_C_SOURCE_PREFIX_FALLBACK = env_nonempty("MODE_C_SOURCE_PREFIX_FALLBACK", "src")


def now_utc():
    return datetime.now(timezone.utc)


def iso_now_full():
    return now_utc().isoformat()


def r2():
    if not AWS_ACCESS_KEY_ID:
        raise RuntimeError("Falta AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("Falta AWS_SECRET_ACCESS_KEY")
    if not R2_ENDPOINT_URL:
        raise RuntimeError("Falta R2_ENDPOINT_URL")
    if not BUCKET_NAME:
        raise RuntimeError("Falta BUCKET_NAME")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def safe_json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def short_hash_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def short_hash_bytes(data):
    return hashlib.sha1(data).hexdigest()[:10]


def safe_slug(text):
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\-_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] or "video"


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

    if "generated_clips" not in state or not isinstance(state["generated_clips"], list):
        state["generated_clips"] = []

    return state


def save_state(state):
    if not isinstance(state, dict):
        state = {}

    if "processed" not in state or not isinstance(state["processed"], list):
        state["processed"] = []

    if "generated_clips" not in state or not isinstance(state["generated_clips"], list):
        state["generated_clips"] = []

    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=STATE_KEY,
        Body=safe_json_dumps(state),
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
    return auto_videos + manual_videos


def download(key, path):
    r2().download_file(
        BUCKET_NAME,
        key,
        path
    )


def upload_video(path, key):
    r2().upload_file(
        path,
        BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": "video/mp4"}
    )


def upload_json(payload, key):
    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=safe_json_dumps(payload),
        ContentType="application/json",
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
        "-ss", str(round(start, 3)),
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
        ("vct", "valorant"),

        ("cs2", "cs2"),
        ("counter strike", "cs2"),
        ("counter-strike", "cs2"),
        ("starladder", "cs2"),
        ("blast premier", "cs2"),
        ("pgl", "cs2"),
        ("major", "cs2"),

        ("league of legends", "leagueoflegends"),
        ("lck", "leagueoflegends"),
        ("lec", "leagueoflegends"),
        ("lcs", "leagueoflegends"),
        ("lpl", "leagueoflegends"),
        ("worlds", "leagueoflegends"),
        ("lol", "leagueoflegends"),

        ("fortnite", "fortnite"),
        ("fncs", "fortnite"),
        ("bugha", "fortnite"),

        ("warzone", "warzone"),
        ("call of duty", "warzone"),
        ("verdansk", "warzone"),
        ("rebirth island", "warzone"),

        ("apex legends", "apex"),
        ("algs", "apex"),
        ("final circles", "apex"),
        ("imperialhal", "apex"),

        ("minecraft", "minecraft"),
        ("hardcore", "minecraft"),
        ("speedrun", "minecraft"),
        ("bedwars", "minecraft"),

        ("ea sports fc", "easportsfc"),
        ("fc pro", "easportsfc"),
        ("echampionsleague", "easportsfc"),
        ("eeuro", "easportsfc"),
        ("vejrgang", "easportsfc"),
        ("tekkz", "easportsfc"),

        ("f1", "f1"),
        ("sim racing", "f1"),
        ("formula 1", "f1"),
        ("jarno opmeer", "f1"),

        ("gran turismo", "granturismo"),
        ("gt world series", "granturismo"),
        ("gt sport", "granturismo"),
    ]

    for needle, label in checks:
        if needle in text:
            return label

    return "generic"


def extract_source_video_id(source_key):
    base = os.path.basename(source_key).rsplit(".", 1)[0]
    base = safe_slug(base)

    # intenta extraer ids conocidos
    m = re.search(r"__youtube__([a-zA-Z0-9_-]{6,})__", source_key)
    if m:
        return f"youtube_{m.group(1)}"

    m = re.search(r"__twitch__([a-zA-Z0-9_-]{6,})__", source_key)
    if m:
        return f"twitch_{m.group(1)}"

    return f"{MODE_C_SOURCE_PREFIX_FALLBACK}_{base[:80]}"


def choose_candidate_starts(duration, clip_seconds, max_clips):
    starts = []

    if duration <= clip_seconds + 1:
        return starts

    start_padding = MODE_C_START_PADDING_SECONDS
    end_padding = MODE_C_END_PADDING_SECONDS
    min_gap = MODE_C_MIN_GAP_SECONDS

    # si es video corto, relaja reglas
    if duration <= MODE_C_SHORT_VIDEO_THRESHOLD:
        start_padding = min(2.0, max(0.0, duration * 0.05))
        end_padding = min(2.0, max(0.0, duration * 0.05))
        min_gap = max(float(clip_seconds), 8.0)

    min_start = max(0.0, start_padding)
    max_start = max(min_start, duration - clip_seconds - end_padding)

    if max_start <= min_start:
        min_start = 0.0
        max_start = max(0.0, duration - clip_seconds)

    if max_start <= min_start:
        return starts

    attempts = 0
    max_attempts = max_clips * 20

    while len(starts) < max_clips and attempts < max_attempts:
        attempts += 1
        candidate = round(random.uniform(min_start, max_start), 3)

        too_close = False
        for s in starts:
            if abs(candidate - s) < min_gap:
                too_close = True
                break

        if too_close:
            continue

        starts.append(candidate)

    starts.sort()
    return starts


def candidate_score(duration, start, clip_seconds):
    """
    Score simple, placeholder útil:
    - penaliza inicio extremo
    - penaliza final extremo
    - favorece zona media
    """
    if duration <= 0:
        return 0.0

    center = start + (clip_seconds / 2.0)
    relative = center / max(duration, 1.0)

    # ideal alrededor del medio del video
    distance_to_mid = abs(relative - 0.5)
    score = max(0.0, 1.0 - (distance_to_mid * 1.6))

    return round(score, 4)


def build_clip_base_name(game_slug, source_video_id, clip_index, start, end):
    start_tag = f"{int(round(start))}s"
    end_tag = f"{int(round(end))}s"
    return f"{game_slug}__{source_video_id}__c{clip_index}__{start_tag}_{end_tag}"


def build_clip_key(base_name):
    return f"{OUTPUT_PREFIX}/{base_name}.mp4"


def build_meta_key(base_name):
    return f"{OUTPUT_META_PREFIX}/{base_name}.json"


def build_clip_meta(
    source_key,
    source_video_id,
    game_slug,
    clip_index,
    start,
    duration,
    clip_key,
    clip_hash,
    score,
):
    end = round(start + duration, 3)
    clip_id_raw = f"{source_video_id}|{clip_index}|{round(start,3)}|{duration}"
    clip_id = short_hash_text(clip_id_raw)

    return {
        "clip_id": clip_id,
        "source_video_key": source_key,
        "source_video_id": source_video_id,
        "source_group": source_video_id,
        "game": game_slug,
        "clip_index": clip_index,
        "start": round(start, 3),
        "duration": duration,
        "end": end,
        "candidate_score": score,
        "status": "candidate",
        "created_at": iso_now_full(),
        "clip_key": clip_key,
        "clip_hash": clip_hash,
        "emotion": None,
        "intensity": None,
        "angle": None,
        "caption": None,
    }


def run_mode_c():
    print("===== UGC MODE C START =====")
    print("INPUT_PREFIX:", INPUT_PREFIX)
    print("INPUT_MANUAL_PREFIX:", INPUT_MANUAL_PREFIX)
    print("OUTPUT_PREFIX:", OUTPUT_PREFIX)
    print("OUTPUT_META_PREFIX:", OUTPUT_META_PREFIX)
    print("STATE_KEY:", STATE_KEY)
    print("CLIP_SECONDS:", CLIP_SECONDS)
    print("MAX_INPUTS:", MAX_INPUTS)
    print("MAX_CLIPS_PER_VIDEO:", MAX_CLIPS_PER_VIDEO)
    print("MODE_C_MIN_GAP_SECONDS:", MODE_C_MIN_GAP_SECONDS)

    state = load_state()
    processed = set(state["processed"])

    videos = list_all_input_videos()

    print("Videos totales encontrados:", len(videos))
    print("Videos procesados en state:", len(processed))

    count = 0
    generated_clips = state.get("generated_clips", [])

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

            game_slug = detect_game_from_key(key)
            source_video_id = extract_source_video_id(key)
            starts = choose_candidate_starts(duration, CLIP_SECONDS, MAX_CLIPS_PER_VIDEO)

            print("GAME DETECTED:", game_slug)
            print("SOURCE VIDEO ID:", source_video_id)
            print("STARTS ELEGIDOS:", starts)

            created_for_video = 0

            for i, start in enumerate(starts):
                out = os.path.join(tmp, f"clip{i}.mp4")

                ok = cut_clip(src, start, CLIP_SECONDS, out)
                if not ok:
                    print("ERROR creando clip:", key, "clip", i)
                    continue

                if not os.path.exists(out) or os.path.getsize(out) == 0:
                    print("ERROR clip vacío:", key, "clip", i)
                    continue

                with open(out, "rb") as f:
                    clip_bytes = f.read()

                if not clip_bytes:
                    print("ERROR bytes vacíos:", key, "clip", i)
                    continue

                end = round(start + CLIP_SECONDS, 3)
                base_name = build_clip_base_name(
                    game_slug=game_slug,
                    source_video_id=source_video_id,
                    clip_index=i,
                    start=start,
                    end=end,
                )

                clip_key = build_clip_key(base_name)
                meta_key = build_meta_key(base_name)
                clip_hash = short_hash_bytes(clip_bytes)
                score = candidate_score(duration, start, CLIP_SECONDS)

                meta = build_clip_meta(
                    source_key=key,
                    source_video_id=source_video_id,
                    game_slug=game_slug,
                    clip_index=i,
                    start=start,
                    duration=CLIP_SECONDS,
                    clip_key=clip_key,
                    clip_hash=clip_hash,
                    score=score,
                )

                upload_video(out, clip_key)
                upload_json(meta, meta_key)

                print("Clip creado:", clip_key)
                print("Meta creada:", meta_key)
                print("Score:", score)

                generated_clips.append({
                    "source_video_key": key,
                    "source_video_id": source_video_id,
                    "clip_key": clip_key,
                    "meta_key": meta_key,
                    "created_at": iso_now_full(),
                })

                created_for_video += 1

            if created_for_video == 0:
                print("No se creó ningún clip válido para:", key)

        processed.add(key)
        count += 1

    state["processed"] = list(processed)[-5000:]
    state["generated_clips"] = generated_clips[-10000:]
    save_state(state)

    print("===== MODE C DONE =====")


if __name__ == "__main__":
    run_mode_c()

# ===== FIN: ugc_mode_c.py =====
