# ugc_mode_c.py
import os
import json
import subprocess
import tempfile
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

from ugc_mode_b import (
    r2_client,
    s3_get_bytes,
    s3_put_bytes,
    r2_public_url,
    openai_text,
    ig_publish,
    fb_publish,
    tiktok_publish,
    youtube_publish,
)

BUCKET_NAME = os.getenv("BUCKET_NAME")

LIBRARY_PREFIX = (os.getenv("UGC_LIBRARY_PREFIX") or "ugc/library/raw/").strip()
if not LIBRARY_PREFIX.endswith("/"):
    LIBRARY_PREFIX += "/"

CLIPS_PREFIX = (os.getenv("UGC_CLIPS_PREFIX") or "ugc/library/clips/").strip()
if not CLIPS_PREFIX.endswith("/"):
    CLIPS_PREFIX += "/"

META_PREFIX = (os.getenv("UGC_META_PREFIX") or "ugc/meta/").strip()
if not META_PREFIX.endswith("/"):
    META_PREFIX += "/"

STATE_KEY = (os.getenv("UGC_MODE_C_STATE_KEY") or "ugc/state/mode_c_state.json").strip()

MODE_C_CLIP_SECONDS = int(os.getenv("MODE_C_CLIP_SECONDS", "8"))
MODE_C_MAX_INPUTS = int(os.getenv("MODE_C_MAX_INPUTS", "5"))
MODE_C_MAX_CLIPS_PER_VIDEO = int(os.getenv("MODE_C_MAX_CLIPS_PER_VIDEO", "3"))

MODE_C_PUBLISH_IG = (os.getenv("MODE_C_PUBLISH_IG", "true").strip().lower() in ("1", "true", "yes", "on"))
MODE_C_PUBLISH_FB = (os.getenv("MODE_C_PUBLISH_FB", "true").strip().lower() in ("1", "true", "yes", "on"))
MODE_C_PUBLISH_TIKTOK = (os.getenv("MODE_C_PUBLISH_TIKTOK", "true").strip().lower() in ("1", "true", "yes", "on"))
MODE_C_PUBLISH_YOUTUBE = (os.getenv("MODE_C_PUBLISH_YOUTUBE", "false").strip().lower() in ("1", "true", "yes", "on"))
MODE_C_UPLOAD_ONLY = (os.getenv("MODE_C_UPLOAD_ONLY", "false").strip().lower() in ("1", "true", "yes", "on"))

MODE_C_MIN_SOURCE_BYTES = int(os.getenv("MODE_C_MIN_SOURCE_BYTES", "200000"))


# ------------------------
# helpers
# ------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def safe_slug(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (s or "").strip())
    out = "_".join(part for part in out.split("_") if part)
    return out[:120] or "clip"


def safe_json_bytes(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def s3_get_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        s3 = r2_client()
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    s3_put_bytes(key, safe_json_bytes(payload), "application/json")


def load_state() -> Dict[str, Any]:
    st = s3_get_json(STATE_KEY)
    if not st:
        st = {
            "processed_source_keys": [],
            "processed_clip_keys": [],
            "last_run_at": None,
        }
    st.setdefault("processed_source_keys", [])
    st.setdefault("processed_clip_keys", [])
    st.setdefault("last_run_at", None)
    return st


def save_state(st: Dict[str, Any]) -> None:
    st["last_run_at"] = now_utc_iso()
    st["processed_source_keys"] = st.get("processed_source_keys", [])[-2000:]
    st["processed_clip_keys"] = st.get("processed_clip_keys", [])[-5000:]
    s3_put_json(STATE_KEY, st)


# ------------------------
# ffmpeg helpers
# ------------------------

def get_video_duration_seconds(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = (p.stdout or "").strip()
    if not raw:
        raise RuntimeError("ffprobe no devolvió duración")
    return float(raw)


def make_clip(input_path: str, output_path: str, start: float, duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(max(0, start)),
        "-i", input_path,
        "-t", str(max(1, duration)),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def build_clip_positions(duration: float, clip_seconds: int, max_clips: int) -> List[Tuple[float, float]]:
    if duration <= clip_seconds:
        return [(0.0, max(1.0, duration))]

    positions: List[Tuple[float, float]] = []

    candidates = [
        0.0,
        max(0.0, (duration / 3.0) - (clip_seconds / 2.0)),
        max(0.0, (duration * 0.60) - (clip_seconds / 2.0)),
        max(0.0, duration - clip_seconds),
    ]

    seen = set()
    for start in candidates:
        start_rounded = round(start, 2)
        if start_rounded in seen:
            continue
        seen.add(start_rounded)

        dur = min(float(clip_seconds), max(1.0, duration - start_rounded))
        if dur < 2:
            continue

        positions.append((start_rounded, dur))
        if len(positions) >= max_clips:
            break

    return positions


# ------------------------
# captions
# ------------------------

def generate_caption(filename: str, hook_type: str) -> str:
    prompt = f"""
Eres editor viral de esports y gaming LATAM.

Tipo de hook: {hook_type}

Genera un caption corto para reel gaming polémico.

Reglas:
- tono fuerte pero no tóxico
- máximo 5 líneas
- incluir pregunta final
- incluir hashtags gaming/esports
- sonar comentable y rápido

Archivo: {filename}
"""
    try:
        text = openai_text(prompt).strip()
        return text or f"Esto va a dividir a la comunidad.\n\n¿Tú qué opinas?\n#gaming #esports"
    except Exception:
        return "Esto va a dividir a la comunidad.\n\n¿Tú qué opinas?\n#gaming #esports"


# ------------------------
# publishing
# ------------------------

def publish_clip(url: str, local_video_path: str, caption: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ig": None,
        "fb": None,
        "tiktok": None,
        "youtube": None,
    }

    if MODE_C_UPLOAD_ONLY:
        return result

    if MODE_C_PUBLISH_IG:
        try:
            result["ig"] = ig_publish(url, caption)
        except Exception as e:
            result["ig"] = {"ok": False, "error": str(e)}

    if MODE_C_PUBLISH_FB:
        try:
            result["fb"] = fb_publish(url, caption)
        except Exception as e:
            result["fb"] = {"ok": False, "error": str(e)}

    if MODE_C_PUBLISH_TIKTOK:
        try:
            result["tiktok"] = tiktok_publish(url, caption)
        except Exception as e:
            result["tiktok"] = {"ok": False, "error": str(e)}

    if MODE_C_PUBLISH_YOUTUBE:
        try:
            result["youtube"] = youtube_publish(local_video_path, caption)
        except Exception as e:
            result["youtube"] = {"ok": False, "error": str(e)}

    return result


# ------------------------
# main
# ------------------------

def run_mode_c():
    print("===== UGC MODE C START =====")
    print("LIBRARY_PREFIX:", LIBRARY_PREFIX)
    print("CLIPS_PREFIX:", CLIPS_PREFIX)
    print("META_PREFIX:", META_PREFIX)
    print("STATE_KEY:", STATE_KEY)
    print("MODE_C_CLIP_SECONDS:", MODE_C_CLIP_SECONDS)
    print("MODE_C_MAX_INPUTS:", MODE_C_MAX_INPUTS)
    print("MODE_C_MAX_CLIPS_PER_VIDEO:", MODE_C_MAX_CLIPS_PER_VIDEO)
    print("MODE_C_UPLOAD_ONLY:", MODE_C_UPLOAD_ONLY)
    print("MODE_C_PUBLISH_IG:", MODE_C_PUBLISH_IG)
    print("MODE_C_PUBLISH_FB:", MODE_C_PUBLISH_FB)
    print("MODE_C_PUBLISH_TIKTOK:", MODE_C_PUBLISH_TIKTOK)
    print("MODE_C_PUBLISH_YOUTUBE:", MODE_C_PUBLISH_YOUTUBE)

    state = load_state()
    processed_source_keys = set(state.get("processed_source_keys", []))
    processed_clip_keys = set(state.get("processed_clip_keys", []))

    s3 = r2_client()
    objs = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=LIBRARY_PREFIX).get("Contents", [])

    source_keys = []
    for obj in objs:
        key = obj.get("Key", "")
        size = int(obj.get("Size", 0) or 0)

        if not key.endswith(".mp4"):
            continue
        if key in processed_source_keys:
            continue
        if size < MODE_C_MIN_SOURCE_BYTES:
            print("SKIP source too small:", key, "| bytes:", size)
            continue

        source_keys.append(key)

    source_keys = source_keys[:MODE_C_MAX_INPUTS]
    print("Videos fuente a procesar:", len(source_keys))

    total_clips_uploaded = 0

    for key in source_keys:
        print("\nAnalizando fuente:", key)

        try:
            video_bytes = s3_get_bytes(key)
            if not video_bytes or len(video_bytes) < MODE_C_MIN_SOURCE_BYTES:
                print("SKIP fuente inválida o muy pequeña:", key)
                continue
        except Exception as e:
            print("No se pudo descargar fuente:", key, "|", str(e))
            continue

        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "video.mp4")

            with open(input_path, "wb") as f:
                f.write(video_bytes)

            try:
                duration = get_video_duration_seconds(input_path)
            except Exception as e:
                print("No se pudo leer duración:", key, "|", str(e))
                continue

            print("Duración detectada:", round(duration, 2), "seg")

            clips = build_clip_positions(
                duration=duration,
                clip_seconds=MODE_C_CLIP_SECONDS,
                max_clips=MODE_C_MAX_CLIPS_PER_VIDEO,
            )

            if not clips:
                print("No se generaron posiciones válidas:", key)
                continue

            hooks = ["drama", "controversia", "hype", "opinion", "debate"]

            for i, clip in enumerate(clips):
                start, dur = clip
                output = os.path.join(td, f"clip{i}.mp4")
                hook_type = hooks[i % len(hooks)]

                print(f"Generando clip #{i} | start={start} dur={dur} | hook={hook_type}")

                try:
                    make_clip(input_path, output, start, dur)
                except Exception as e:
                    print("FFmpeg falló en clip:", key, "|", i, "|", str(e))
                    continue

                try:
                    with open(output, "rb") as f:
                        clip_bytes = f.read()
                except Exception as e:
                    print("No se pudo leer clip generado:", str(e))
                    continue

                if not clip_bytes or len(clip_bytes) < 50_000:
                    print("Clip generado demasiado pequeño, skip")
                    continue

                source_base = os.path.basename(key).rsplit(".", 1)[0]
                clip_name = f"{safe_slug(source_base)}__{i}__{short_hash(key + str(i))}.mp4"
                clip_key = f"{CLIPS_PREFIX}{clip_name}"

                if clip_key in processed_clip_keys:
                    print("SKIP clip ya procesado:", clip_key)
                    continue

                caption = generate_caption(key, hook_type)

                try:
                    s3_put_bytes(clip_key, clip_bytes, "video/mp4")
                    url = r2_public_url(clip_key)
                except Exception as e:
                    print("No se pudo subir clip a R2:", clip_key, "|", str(e))
                    continue

                publish_res = publish_clip(url, output, caption)

                meta = {
                    "source_video_key": key,
                    "clip_key": clip_key,
                    "clip_url": url,
                    "caption": caption,
                    "hook_type": hook_type,
                    "clip_index": i,
                    "start_seconds": start,
                    "duration_seconds": dur,
                    "source_duration_seconds": duration,
                    "generated_at": now_utc_iso(),
                    "publish_result": publish_res,
                }

                meta_key = f"{META_PREFIX}{os.path.basename(clip_key).rsplit('.', 1)[0]}.json"

                try:
                    s3_put_json(meta_key, meta)
                except Exception as e:
                    print("No se pudo guardar meta:", meta_key, "|", str(e))

                processed_clip_keys.add(clip_key)
                total_clips_uploaded += 1

                print("OK clip:", url)

        processed_source_keys.add(key)

    state["processed_source_keys"] = list(processed_source_keys)
    state["processed_clip_keys"] = list(processed_clip_keys)
    save_state(state)

    print("\n===== UGC MODE C DONE =====")
    print("Total clips subidos:", total_clips_uploaded)


if __name__ == "__main__":
    run_mode_c()
