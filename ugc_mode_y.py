# ugc_mode_y.py
import os
import re
import json
import hashlib
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import boto3


# =========================
# ENV helpers
# =========================

def env_nonempty(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


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


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# =========================
# Config
# =========================

YOUTUBE_SEARCH_TERMS = env_nonempty(
    "YOUTUBE_SEARCH_TERMS",
    "EA Sports FC,F1,Gran Turismo,VALORANT,CS2,League of Legends,Fortnite,Warzone,Apex Legends,Minecraft",
)

YOUTUBE_MAX_RESULTS_PER_TERM = env_int("YOUTUBE_MAX_RESULTS_PER_TERM", 5)
YOUTUBE_MAX_DOWNLOADS_PER_RUN = env_int("YOUTUBE_MAX_DOWNLOADS_PER_RUN", 3)
YOUTUBE_SEARCH_DAYS = env_int("YOUTUBE_SEARCH_DAYS", 7)
YOUTUBE_MIN_VIDEO_BYTES = env_int("YOUTUBE_MIN_VIDEO_BYTES", 500_000)
YOUTUBE_MIN_DURATION_SEC = env_float("YOUTUBE_MIN_DURATION_SEC", 30.0)
YOUTUBE_MAX_DURATION_SEC = env_float("YOUTUBE_MAX_DURATION_SEC", 14_400.0)
YOUTUBE_ONLY_LIVE_REPLAYS = env_bool("YOUTUBE_ONLY_LIVE_REPLAYS", False)

STATE_KEY = env_nonempty("YOUTUBE_MODE_Y_STATE_KEY", "ugc/state/mode_y_state.json")
R2_META_PREFIX = (env_nonempty("UGC_META_PREFIX", "ugc/meta/") or "ugc/meta/").strip()
if not R2_META_PREFIX.endswith("/"):
    R2_META_PREFIX += "/"

R2_INBOX_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or "").rstrip("/")

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

YT_DLP_BIN = env_nonempty("YT_DLP_BIN", "yt-dlp") or "yt-dlp"
YT_DLP_COOKIES_FILE = env_nonempty("YT_DLP_COOKIES_FILE")


# =========================
# Generic helpers
# =========================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().strftime("%Y-%m-%d")


def iso_now_full() -> str:
    return now_utc().isoformat()


def safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9\-_]+", "_", s)[:120]
    return s.strip("_") or "video"


def short_hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:10]


def safe_json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout or "", p.stderr or ""


def ffprobe_json(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    code, stdout, _ = run_cmd(cmd)
    if code != 0:
        return {}
    try:
        return json.loads(stdout or "{}")
    except Exception:
        return {}


def get_video_duration_seconds(path: str) -> float:
    info = ffprobe_json(path)
    try:
        return float(info.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        return 0.0


def get_file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def is_probably_mp4_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        return b"ftyp" in head
    except Exception:
        return False


# =========================
# R2 helpers
# =========================

def r2_client():
    if not (R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and BUCKET_NAME):
        raise RuntimeError("Faltan credenciales R2/S3")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def s3_put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    s3 = r2_client()
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    if R2_PUBLIC_BASE_URL.startswith("http"):
        return f"{R2_PUBLIC_BASE_URL}/{key}"
    return key


def s3_put_json(key: str, payload: Dict[str, Any]):
    s3_put_bytes(key, safe_json_dumps(payload), "application/json")


def s3_get_json(key: str):
    try:
        obj = r2_client().get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


# =========================
# State
# =========================

def load_state() -> Dict[str, Any]:
    st = s3_get_json(STATE_KEY)
    if not st:
        st = {"processed_video_ids": [], "last_run_at": None}
    st.setdefault("processed_video_ids", [])
    st.setdefault("last_run_at", None)
    return st


def save_state(st: Dict[str, Any]):
    st["last_run_at"] = iso_now_full()
    s3_put_json(STATE_KEY, st)


# =========================
# yt-dlp search / download
# =========================

def yt_dlp_base_args() -> List[str]:
    args = [
        YT_DLP_BIN,
        "--remote-components", "ejs:github",
    ]
    if YT_DLP_COOKIES_FILE:
        args += ["--cookies", YT_DLP_COOKIES_FILE]
    return args


def build_search_query(term: str) -> str:
    base = term.strip()
    if YOUTUBE_ONLY_LIVE_REPLAYS:
        return f"ytsearch{YOUTUBE_MAX_RESULTS_PER_TERM}:{base} live stream replay"
    return f"ytsearch{YOUTUBE_MAX_RESULTS_PER_TERM}:{base} highlights gameplay esports"


def yt_dlp_search(term: str) -> List[Dict[str, Any]]:
    query = build_search_query(term)
    cmd = yt_dlp_base_args() + [
        "--dump-single-json",
        "--skip-download",
        "--dateafter", f"today-{YOUTUBE_SEARCH_DAYS}days",
        query,
    ]

    code, stdout, stderr = run_cmd(cmd)
    if code != 0:
        print("yt-dlp search error:", stderr[:1000])
        return []

    try:
        data = json.loads(stdout)
    except Exception:
        return []

    entries = data.get("entries") or []
    out: List[Dict[str, Any]] = []

    for e in entries:
        if not e:
            continue

        video_id = e.get("id")
        title = e.get("title") or ""
        webpage_url = e.get("webpage_url") or e.get("original_url") or ""
        channel = e.get("channel") or e.get("uploader") or ""
        duration = float(e.get("duration") or 0.0)
        view_count = int(e.get("view_count") or 0)
        was_live = bool(e.get("was_live") or False)
        live_status = e.get("live_status") or ""

        if not video_id or not webpage_url:
            continue
        if duration and duration < YOUTUBE_MIN_DURATION_SEC:
            continue
        if duration and duration > YOUTUBE_MAX_DURATION_SEC:
            continue

        out.append(
            {
                "id": video_id,
                "title": title,
                "url": webpage_url,
                "channel": channel,
                "duration": duration,
                "view_count": view_count,
                "was_live": was_live,
                "live_status": live_status,
                "search_term": term,
            }
        )

    return out


def youtube_score(item: Dict[str, Any]) -> float:
    views = int(item.get("view_count") or 0)
    duration = float(item.get("duration") or 0.0)
    title = (item.get("title") or "").lower()
    search_term = (item.get("search_term") or "").lower()

    bonus = 0
    hot_words = [
        "highlights", "gameplay", "best moments", "clutch", "ace",
        "ranked", "pro", "esports", "insane", "final", "goals", "overtake"
    ]
    for w in hot_words:
        if w in title:
            bonus += 120

    if "valorant" in search_term or "cs2" in search_term or "league of legends" in search_term:
        bonus += 80
    if "ea sports fc" in search_term or "f1" in search_term or "gran turismo" in search_term:
        bonus += 60

    duration_bonus = 100 if 60 <= duration <= 1800 else 0
    return (views * 0.05) + bonus + duration_bonus


def yt_dlp_download_video(video_url: str, out_path: str) -> bool:
    cmd = yt_dlp_base_args() + [
        "-f", "mp4/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "-o", out_path,
        video_url,
    ]

    code, _, stderr = run_cmd(cmd)
    if code != 0:
        print("yt-dlp download error:", stderr[:1000])
        return False
    return True


# =========================
# Validation
# =========================

def validate_downloaded_video(path: str) -> Tuple[bool, str]:
    size = get_file_size_bytes(path)
    duration = get_video_duration_seconds(path)

    print("Downloaded file size:", size)
    print("Downloaded duration:", duration)

    if size < YOUTUBE_MIN_VIDEO_BYTES:
        return False, f"too_small:{size}"
    if duration < YOUTUBE_MIN_DURATION_SEC:
        return False, f"too_short:{duration}"
    if duration > YOUTUBE_MAX_DURATION_SEC:
        return False, f"too_long:{duration}"
    if not is_probably_mp4_file(path):
        return False, "missing_ftyp"

    return True, "ok"


# =========================
# Main
# =========================

def run_mode_y() -> None:
    print("===== MODE Y (YOUTUBE GAMER HARVESTER) -> R2 INBOX =====")
    print("YOUTUBE_SEARCH_TERMS:", YOUTUBE_SEARCH_TERMS)
    print("YOUTUBE_MAX_RESULTS_PER_TERM:", YOUTUBE_MAX_RESULTS_PER_TERM)
    print("YOUTUBE_MAX_DOWNLOADS_PER_RUN:", YOUTUBE_MAX_DOWNLOADS_PER_RUN)
    print("YOUTUBE_SEARCH_DAYS:", YOUTUBE_SEARCH_DAYS)
    print("YOUTUBE_MIN_VIDEO_BYTES:", YOUTUBE_MIN_VIDEO_BYTES)
    print("YOUTUBE_MIN_DURATION_SEC:", YOUTUBE_MIN_DURATION_SEC)
    print("YOUTUBE_MAX_DURATION_SEC:", YOUTUBE_MAX_DURATION_SEC)
    print("STATE_KEY:", STATE_KEY)
    print("YT_DLP_COOKIES_FILE set:", bool(YT_DLP_COOKIES_FILE))

    terms = [x.strip() for x in (YOUTUBE_SEARCH_TERMS or "").split(",") if x.strip()]
    if not terms:
        raise RuntimeError("YOUTUBE_SEARCH_TERMS vacío")

    state = load_state()
    processed_ids = set(state.get("processed_video_ids", []))
    print("Processed video ids in state:", len(processed_ids))

    all_candidates: List[Tuple[float, Dict[str, Any]]] = []

    for term in terms:
        print("Searching YouTube:", term)
        found = yt_dlp_search(term)
        print(f"Found {len(found)} candidates for {term}")

        for item in found:
            vid = item.get("id")
            title = item.get("title") or ""

            if not vid:
                continue
            if vid in processed_ids:
                print("SKIP already processed:", vid, "|", title[:100])
                continue

            score = youtube_score(item)
            print("CANDIDATE:", vid, "| score:", round(score, 2), "|", title[:100])
            all_candidates.append((score, item))

    if not all_candidates:
        print("No hay videos candidatos.")
        return

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    picked = all_candidates[:max(1, YOUTUBE_MAX_DOWNLOADS_PER_RUN)]

    print(f"Seleccionados {len(picked)} videos para bajar a inbox.")

    uploaded_count = 0

    for score, item in picked:
        video_id = item["id"]
        title = item.get("title") or ""
        url = item.get("url") or ""
        channel = item.get("channel") or ""
        search_term = item.get("search_term") or ""

        with tempfile.TemporaryDirectory() as td:
            out_template = os.path.join(td, "video.%(ext)s")
            target_mp4 = os.path.join(td, "video.mp4")

            print("Downloading:", url)
            ok = yt_dlp_download_video(url, out_template)
            if not ok:
                continue

            downloaded = None
            for name in os.listdir(td):
                if name.startswith("video.") and name.lower().endswith(".mp4"):
                    downloaded = os.path.join(td, name)
                    break

            if not downloaded:
                print("Download error: no se encontró mp4 descargado")
                continue

            if downloaded != target_mp4:
                os.replace(downloaded, target_mp4)

            valid, reason = validate_downloaded_video(target_mp4)
            if not valid:
                print("Video descartado:", reason)
                continue

            with open(target_mp4, "rb") as f:
                data = f.read()

        h = short_hash_bytes(data)
        key = f"{R2_INBOX_PREFIX}/{iso_now()}__youtube__{video_id}__{safe_slug(title)}__{h}.mp4"

        print("Uploading to R2 inbox:", key)
        s3_put_bytes(key, data, "video/mp4")

        meta = {
            "source": "youtube",
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "url": url,
            "search_term": search_term,
            "duration": item.get("duration"),
            "view_count": item.get("view_count"),
            "score": score,
            "queued_at": iso_now_full(),
            "r2_key": key,
        }

        meta_base = os.path.basename(key).rsplit(".", 1)[0] + ".json"
        meta_key = f"{R2_META_PREFIX}{meta_base}"
        s3_put_json(meta_key, meta)

        processed_ids.add(video_id)
        uploaded_count += 1

        print(f"OK: {title[:80]} | score: {round(score, 2)}")
        print("Saved meta:", meta_key)

    state["processed_video_ids"] = list(processed_ids)[-5000:]
    save_state(state)

    print("===== MODE Y DONE =====")
    print("Uploaded valid videos:", uploaded_count)


if __name__ == "__main__":
    run_mode_y()
