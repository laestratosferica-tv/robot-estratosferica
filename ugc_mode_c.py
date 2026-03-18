# ===== INICIO: ugc_mode_c.py =====

import os
import re
import json
import math
import hashlib
import subprocess
import tempfile
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


AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

INPUT_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")
INPUT_MANUAL_PREFIX = (env_nonempty("UGC_INBOX_MANUAL_PREFIX", "ugc/inbox_manual") or "ugc/inbox_manual").strip().strip("/")
OUTPUT_PREFIX = (env_nonempty("UGC_CLIPS_PREFIX", "ugc/library/clips") or "ugc/library/clips").strip().strip("/")
META_CLIPS_PREFIX = (env_nonempty("UGC_META_CLIPS_PREFIX", "ugc/meta/clips") or "ugc/meta/clips").strip().strip("/")

STATE_KEY = env_nonempty("MODE_C_STATE_KEY", "ugc/state/mode_c_state.json")

# NUEVO: duración flexible
CLIP_SECONDS_DEFAULT = env_float("MODE_C_CLIP_SECONDS_DEFAULT", 12.0)
CLIP_SECONDS_MIN = env_float("MODE_C_CLIP_SECONDS_MIN", 8.0)
CLIP_SECONDS_MAX = env_float("MODE_C_CLIP_SECONDS_MAX", 15.0)

MOMENT_DURATION_CLUTCH = env_float("MODE_C_MOMENT_DURATION_CLUTCH", 14.0)
MOMENT_DURATION_REACTION = env_float("MODE_C_MOMENT_DURATION_REACTION", 10.0)
MOMENT_DURATION_ACTION = env_float("MODE_C_MOMENT_DURATION_ACTION", 9.0)
MOMENT_DURATION_HYPE = env_float("MODE_C_MOMENT_DURATION_HYPE", 12.0)
MOMENT_DURATION_MOMENT = env_float("MODE_C_MOMENT_DURATION_MOMENT", 11.0)

MAX_INPUTS = env_int("MODE_C_MAX_INPUTS", 4)
MAX_CLIPS_PER_VIDEO = env_int("MODE_C_MAX_CLIPS_PER_VIDEO", 4)

ANALYSIS_STEP_SEC = env_float("MODE_C_ANALYSIS_STEP_SEC", 1.0)
MIN_GAP_BETWEEN_CLIPS_SEC = env_float("MODE_C_MIN_GAP_BETWEEN_CLIPS_SEC", 18.0)
MIN_SOURCE_DURATION_SEC = env_float("MODE_C_MIN_SOURCE_DURATION_SEC", 25.0)

AUDIO_WEIGHT = env_float("MODE_C_AUDIO_WEIGHT", 0.55)
VIDEO_WEIGHT = env_float("MODE_C_VIDEO_WEIGHT", 0.45)

EDGE_PADDING_SEC = env_float("MODE_C_EDGE_PADDING_SEC", 3.0)

FFMPEG_TIMEOUT_SEC = env_int("MODE_C_FFMPEG_TIMEOUT_SEC", 1800)
FFPROBE_TIMEOUT_SEC = env_int("MODE_C_FFPROBE_TIMEOUT_SEC", 180)
AUDIO_ANALYSIS_HARD_LIMIT_SEC = env_float("MODE_C_AUDIO_ANALYSIS_HARD_LIMIT_SEC", 1800.0)
VIDEO_ANALYSIS_HARD_LIMIT_SEC = env_float("MODE_C_VIDEO_ANALYSIS_HARD_LIMIT_SEC", 1800.0)


def now_utc():
    return datetime.now(timezone.utc)


def iso_now_full():
    return now_utc().isoformat()


def short_hash_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def safe_json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


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


def load_state():
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=STATE_KEY)
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
    r2().download_file(BUCKET_NAME, key, path)


def upload(path, key):
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


def ffprobe_json(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        print("ffprobe timeout:", path)
        return {}
    except Exception as e:
        print("ffprobe exception:", repr(e))
        return {}

    if p.returncode != 0:
        print("ffprobe error:", (p.stderr or "")[:1000])
        return {}

    try:
        return json.loads(p.stdout or "{}")
    except Exception as e:
        print("ffprobe json parse error:", repr(e))
        return {}


def get_duration(path):
    info = ffprobe_json(path)
    try:
        return float(info.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        return 0.0


def cut_clip(src, start, seconds, dst):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(max(0.0, start)),
        "-i", src,
        "-t", str(seconds),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        dst,
    ]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
        if p.returncode != 0:
            print("ERROR ffmpeg cut_clip:", (p.stderr or "")[:1200])
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Timeout en cut_clip:", src)
        return False
    except Exception as e:
        print("Exception en cut_clip:", repr(e))
        return False


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
        ("imperialhal", "apex"),
        ("minecraft", "minecraft"),
        ("bedwars", "minecraft"),
        ("speedrun", "minecraft"),
        ("ea sports fc", "easportsfc"),
        ("fc pro", "easportsfc"),
        ("echampionsleague", "easportsfc"),
        ("vejrgang", "easportsfc"),
        ("tekkz", "easportsfc"),
        ("f1", "f1"),
        ("formula 1", "f1"),
        ("gran turismo", "granturismo"),
        ("gt world series", "granturismo"),
    ]

    for needle, label in checks:
        if needle in text:
            return label

    return "generic"


def safe_source_video_id_from_key(source_key):
    base = os.path.basename(source_key).rsplit(".", 1)[0]
    return short_hash_text(source_key + "::" + base)


def build_clip_key(source_key, clip_index, start_sec, end_sec):
    game_slug = detect_game_from_key(source_key)
    source_video_id = safe_source_video_id_from_key(source_key)
    start_i = int(round(start_sec))
    end_i = int(round(end_sec))
    return f"{OUTPUT_PREFIX}/{game_slug}__{source_video_id}__c{clip_index}__{start_i}s_{end_i}s.mp4"


def build_clip_meta_key_from_clip_key(clip_key):
    base = os.path.basename(clip_key).rsplit(".", 1)[0]
    return f"{META_CLIPS_PREFIX}/{base}.json"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values):
    if not values:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def normalize_series(values):
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def smooth_series(values, radius=1):
    if not values:
        return []
    out = []
    n = len(values)
    for i in range(n):
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        out.append(mean(values[a:b]))
    return out


def extract_audio_energy_series(src_path, duration, step_sec):
    """
    Saca RMS aproximado por ventana usando ffmpeg astats.
    Si falla, devuelve una serie plana.
    """
    analysis_duration = min(duration, AUDIO_ANALYSIS_HARD_LIMIT_SEC)
    with tempfile.TemporaryDirectory() as td:
        null_out = os.path.join(td, "null.wav")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "info",
            "-t", str(analysis_duration),
            "-i", src_path,
            "-af", "asetnsamples=n=44100,astats=metadata=1:reset=1",
            "-f", "null",
            null_out,
        ]
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=FFMPEG_TIMEOUT_SEC,
            )
            stderr = (p.stderr or "") + "\n" + (p.stdout or "")
        except subprocess.TimeoutExpired:
            print("Timeout en extract_audio_energy_series:", src_path)
            count = max(1, int(math.ceil(duration / step_sec)))
            return [0.0] * count
        except Exception as e:
            print("Exception en extract_audio_energy_series:", repr(e))
            count = max(1, int(math.ceil(duration / step_sec)))
            return [0.0] * count

    rms_vals = []
    for line in stderr.splitlines():
        if "RMS level dB" in line:
            try:
                val = float(line.strip().split(":")[-1].strip())
                rms_vals.append(val)
            except Exception:
                pass

    if not rms_vals:
        count = max(1, int(math.ceil(duration / step_sec)))
        return [0.0] * count

    energy = []
    for db in rms_vals:
        energy.append(60.0 + db)

    count = max(1, int(math.ceil(duration / step_sec)))

    if len(energy) == count:
        return energy

    if len(energy) < count:
        last = energy[-1] if energy else 0.0
        return energy + [last] * (count - len(energy))

    out = []
    ratio = len(energy) / float(count)
    for i in range(count):
        a = int(i * ratio)
        b = int((i + 1) * ratio)
        if b <= a:
            b = a + 1
        out.append(mean(energy[a:b]))
    return out


def extract_visual_change_series(src_path, duration, step_sec):
    """
    Usa ffmpeg + select(scene) para detectar cambios de escena.
    Si falla, devuelve serie plana.
    """
    analysis_duration = min(duration, VIDEO_ANALYSIS_HARD_LIMIT_SEC)

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, "frames")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "info",
            "-t", str(analysis_duration),
            "-i", src_path,
            "-vf", f"fps=1/{step_sec},select='gt(scene,0.08)',showinfo",
            "-vsync", "vfr",
            os.path.join(out_dir, "frame_%05d.jpg"),
        ]

        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=FFMPEG_TIMEOUT_SEC,
            )
            stderr = (p.stderr or "") + "\n" + (p.stdout or "")
        except subprocess.TimeoutExpired:
            print("Timeout en extract_visual_change_series:", src_path)
            count = max(1, int(math.ceil(duration / step_sec)))
            return [0.0] * count
        except Exception as e:
            print("Exception en extract_visual_change_series:", repr(e))
            count = max(1, int(math.ceil(duration / step_sec)))
            return [0.0] * count

    count = max(1, int(math.ceil(duration / step_sec)))
    scores = [0.0] * count

    pts_times = []
    for line in stderr.splitlines():
        if "pts_time:" in line:
            m = re.search(r"pts_time:([0-9\.]+)", line)
            if m:
                try:
                    pts_times.append(float(m.group(1)))
                except Exception:
                    pass

    for t in pts_times:
        idx = int(t // step_sec)
        if 0 <= idx < count:
            scores[idx] += 1.0

    return scores


def infer_moment_type(audio_n, video_n, peak_audio, peak_video):
    if peak_audio > 0.78 and peak_video > 0.72:
        return "clutch"
    if peak_audio > 0.82:
        return "reaction"
    if peak_video > 0.82:
        return "action"
    if (audio_n + video_n) / 2.0 > 0.65:
        return "hype"
    return "moment"


def infer_emotion(moment_type, total_score):
    if moment_type == "clutch":
        return "clutch"
    if moment_type == "reaction":
        return "shock"
    if moment_type == "action":
        return "skill"
    if total_score > 0.82:
        return "heroic"
    if total_score > 0.68:
        return "chaos"
    return "skill"


def infer_intensity(score):
    if score >= 0.88:
        return "estratosferico"
    if score >= 0.72:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


# NUEVO: decide duración según tipo de momento
def resolve_clip_seconds(moment_type, emotion, candidate_score):
    moment_type = str(moment_type or "").strip().lower()
    emotion = str(emotion or "").strip().lower()
    score = float(candidate_score or 0.0)

    if moment_type == "clutch":
        secs = MOMENT_DURATION_CLUTCH
    elif moment_type == "reaction":
        secs = MOMENT_DURATION_REACTION
    elif moment_type == "action":
        secs = MOMENT_DURATION_ACTION
    elif moment_type == "hype":
        secs = MOMENT_DURATION_HYPE
    else:
        secs = MOMENT_DURATION_MOMENT

    if emotion in ("heroic", "chaos"):
        secs += 1.0

    if score >= 0.82:
        secs += 1.0
    elif score <= 0.45:
        secs -= 1.0

    return round(clamp(secs, CLIP_SECONDS_MIN, CLIP_SECONDS_MAX), 2)


# NUEVO: ya no usa duración fija
def score_segments(duration, audio_series, video_series, step_sec):
    audio_sm = smooth_series(normalize_series(audio_series), radius=1)
    video_sm = smooth_series(normalize_series(video_series), radius=1)

    count = min(len(audio_sm), len(video_sm))
    candidates = []

    for idx in range(count):
        center = idx * step_sec

        audio_n = audio_sm[idx]
        video_n = video_sm[idx]

        rough_moment_type = infer_moment_type(audio_n, video_n, audio_n, video_n)
        rough_score = (audio_n * AUDIO_WEIGHT) + (video_n * VIDEO_WEIGHT)
        rough_emotion = infer_emotion(rough_moment_type, rough_score)

        clip_seconds = resolve_clip_seconds(rough_moment_type, rough_emotion, rough_score)

        pre_ratio = 0.42
        if rough_moment_type == "clutch":
            pre_ratio = 0.50
        elif rough_moment_type == "reaction":
            pre_ratio = 0.35
        elif rough_moment_type == "action":
            pre_ratio = 0.30
        elif rough_moment_type == "hype":
            pre_ratio = 0.45

        start = center - (clip_seconds * pre_ratio)
        start = clamp(
            start,
            EDGE_PADDING_SEC,
            max(EDGE_PADDING_SEC, duration - clip_seconds - EDGE_PADDING_SEC),
        )
        end = start + clip_seconds

        if end > duration:
            start = max(0.0, duration - clip_seconds)
            end = duration

        a_idx = max(0, int(start // step_sec))
        b_idx = min(count, int(math.ceil(end / step_sec)))
        if b_idx <= a_idx:
            b_idx = min(count, a_idx + 1)

        audio_window = audio_sm[a_idx:b_idx]
        video_window = video_sm[a_idx:b_idx]

        audio_peak = max(audio_window) if audio_window else 0.0
        video_peak = max(video_window) if video_window else 0.0
        audio_avg = mean(audio_window)
        video_avg = mean(video_window)

        total_score = (audio_avg * AUDIO_WEIGHT) + (video_avg * VIDEO_WEIGHT)
        moment_type = infer_moment_type(audio_avg, video_avg, audio_peak, video_peak)
        emotion = infer_emotion(moment_type, total_score)
        intensity = infer_intensity(total_score)

        final_clip_seconds = resolve_clip_seconds(moment_type, emotion, total_score)

        if abs(final_clip_seconds - clip_seconds) > 0.5:
            clip_seconds = final_clip_seconds
            start = center - (clip_seconds * pre_ratio)
            start = clamp(
                start,
                EDGE_PADDING_SEC,
                max(EDGE_PADDING_SEC, duration - clip_seconds - EDGE_PADDING_SEC),
            )
            end = start + clip_seconds

            if end > duration:
                start = max(0.0, duration - clip_seconds)
                end = duration

            a_idx = max(0, int(start // step_sec))
            b_idx = min(count, int(math.ceil(end / step_sec)))
            if b_idx <= a_idx:
                b_idx = min(count, a_idx + 1)

            audio_window = audio_sm[a_idx:b_idx]
            video_window = video_sm[a_idx:b_idx]

            audio_peak = max(audio_window) if audio_window else 0.0
            video_peak = max(video_window) if video_window else 0.0
            audio_avg = mean(audio_window)
            video_avg = mean(video_window)

            total_score = (audio_avg * AUDIO_WEIGHT) + (video_avg * VIDEO_WEIGHT)
            moment_type = infer_moment_type(audio_avg, video_avg, audio_peak, video_peak)
            emotion = infer_emotion(moment_type, total_score)
            intensity = infer_intensity(total_score)

        candidates.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "clip_seconds": round(end - start, 2),
                "audio_score": round(audio_avg, 4),
                "video_score": round(video_avg, 4),
                "audio_peak_score": round(audio_peak, 4),
                "visual_peak_score": round(video_peak, 4),
                "candidate_score": round(total_score, 4),
                "moment_type": moment_type,
                "emotion": emotion,
                "intensity": intensity,
            }
        )

    candidates.sort(key=lambda x: x["candidate_score"], reverse=True)
    return candidates


def pick_diverse_segments(candidates, max_clips, min_gap_sec):
    selected = []

    for c in candidates:
        if len(selected) >= max_clips:
            break

        too_close = False
        for s in selected:
            dynamic_gap = max(min_gap_sec, min(c["duration"], s["duration"]) * 0.8)
            if abs(c["start"] - s["start"]) < dynamic_gap:
                too_close = True
                break

        if too_close:
            continue

        selected.append(c)

    return selected


def run_mode_c():
    print("===== UGC MODE C START =====")
    print("INPUT_PREFIX:", INPUT_PREFIX)
    print("INPUT_MANUAL_PREFIX:", INPUT_MANUAL_PREFIX)
    print("OUTPUT_PREFIX:", OUTPUT_PREFIX)
    print("META_CLIPS_PREFIX:", META_CLIPS_PREFIX)
    print("STATE_KEY:", STATE_KEY)
    print("CLIP_SECONDS_DEFAULT:", CLIP_SECONDS_DEFAULT)
    print("CLIP_SECONDS_MIN:", CLIP_SECONDS_MIN)
    print("CLIP_SECONDS_MAX:", CLIP_SECONDS_MAX)
    print("MOMENT_DURATION_ACTION:", MOMENT_DURATION_ACTION)
    print("MOMENT_DURATION_REACTION:", MOMENT_DURATION_REACTION)
    print("MOMENT_DURATION_HYPE:", MOMENT_DURATION_HYPE)
    print("MOMENT_DURATION_MOMENT:", MOMENT_DURATION_MOMENT)
    print("MOMENT_DURATION_CLUTCH:", MOMENT_DURATION_CLUTCH)
    print("MAX_INPUTS:", MAX_INPUTS)
    print("MAX_CLIPS_PER_VIDEO:", MAX_CLIPS_PER_VIDEO)
    print("ANALYSIS_STEP_SEC:", ANALYSIS_STEP_SEC)
    print("MIN_GAP_BETWEEN_CLIPS_SEC:", MIN_GAP_BETWEEN_CLIPS_SEC)
    print("AUDIO_WEIGHT:", AUDIO_WEIGHT)
    print("VIDEO_WEIGHT:", VIDEO_WEIGHT)

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

            if duration < max(CLIP_SECONDS_MIN + 2.0, MIN_SOURCE_DURATION_SEC):
                print("Video demasiado corto:", key)
                processed.add(key)
                continue

            try:
                audio_series = extract_audio_energy_series(src, duration, ANALYSIS_STEP_SEC)
                video_series = extract_visual_change_series(src, duration, ANALYSIS_STEP_SEC)

                print("Audio samples:", len(audio_series))
                print("Video samples:", len(video_series))

                candidates = score_segments(
                    duration=duration,
                    audio_series=audio_series,
                    video_series=video_series,
                    step_sec=ANALYSIS_STEP_SEC,
                )

                selected = pick_diverse_segments(
                    candidates=candidates,
                    max_clips=MAX_CLIPS_PER_VIDEO,
                    min_gap_sec=MIN_GAP_BETWEEN_CLIPS_SEC,
                )

                print("Candidates analizados:", len(candidates))
                print("Seleccionados:", len(selected))

            except Exception as e:
                print("ERROR analizando video:", key, repr(e))
                processed.add(key)
                continue

            source_video_id = safe_source_video_id_from_key(key)
            source_group = source_video_id
            game_slug = detect_game_from_key(key)

            created_for_video = 0

            for i, seg in enumerate(selected):
                start = float(seg["start"])
                end = float(seg["end"])
                clip_seconds = float(seg.get("duration") or CLIP_SECONDS_DEFAULT)
                out = os.path.join(tmp, f"clip{i}.mp4")

                ok = cut_clip(src, start, clip_seconds, out)
                if not ok:
                    print("ERROR creando clip:", key, "clip", i)
                    continue

                if not os.path.exists(out) or os.path.getsize(out) == 0:
                    print("ERROR clip vacío:", key, "clip", i)
                    continue

                clip_key = build_clip_key(key, i, start, end)
                upload(out, clip_key)

                clip_meta = {
                    "source_video_key": key,
                    "source_video_id": source_video_id,
                    "source_group": source_group,
                    "clip_key": clip_key,
                    "clip_id": short_hash_text(clip_key),
                    "clip_index": i,
                    "game": game_slug,
                    "start": start,
                    "end": end,
                    "duration": seg["duration"],
                    "audio_score": seg["audio_score"],
                    "video_score": seg["video_score"],
                    "audio_peak_score": seg["audio_peak_score"],
                    "visual_peak_score": seg["visual_peak_score"],
                    "candidate_score": seg["candidate_score"],
                    "moment_type": seg["moment_type"],
                    "emotion": seg["emotion"],
                    "intensity": seg["intensity"],
                    "generated_at": iso_now_full(),
                }

                meta_key = build_clip_meta_key_from_clip_key(clip_key)
                upload_json(clip_meta, meta_key)

                print("GAME DETECTED:", game_slug)
                print("Clip creado:", clip_key)
                print("Meta creada:", meta_key)
                print(
                    "Score:", seg["candidate_score"],
                    "| moment:", seg["moment_type"],
                    "| emotion:", seg["emotion"],
                    "| intensity:", seg["intensity"],
                    "| duration:", seg["duration"],
                )

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

# ===== FIN: ugc_mode_c.py =====
