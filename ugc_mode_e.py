# ugc_mode_e.py
import os
import re
import json
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
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
# Twitch
# =========================

TWITCH_CLIENT_ID = env_nonempty("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = env_nonempty("TWITCH_CLIENT_SECRET")

TWITCH_GAMES = env_nonempty(
    "TWITCH_GAMES",
    "Just Chatting,League of Legends,Counter-Strike,Grand Theft Auto V,Minecraft,VALORANT,Apex Legends"
)

TWITCH_CLIPS_PER_GAME = env_int("TWITCH_CLIPS_PER_GAME", 10)
TWITCH_PICK_TOTAL = env_int("TWITCH_PICK_TOTAL", 5)
TWITCH_MIN_VIEWS = env_int("TWITCH_MIN_VIEWS", 800)
TWITCH_PERIOD_DAYS = env_int("TWITCH_PERIOD_DAYS", 7)

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)
USER_AGENT = env_nonempty(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
) or "Mozilla/5.0"

# validación de archivo real
TWITCH_MIN_VIDEO_BYTES = env_int("TWITCH_MIN_VIDEO_BYTES", 200_000)
STRICT_VIDEO_CONTENT_TYPE = env_bool("STRICT_VIDEO_CONTENT_TYPE", False)

# R2 meta/state
STATE_KEY = env_nonempty("TWITCH_MODE_E_STATE_KEY", "ugc/state/mode_e_state.json")
R2_META_PREFIX = (env_nonempty("UGC_META_PREFIX", "ugc/meta/") or "ugc/meta/").strip()
if not R2_META_PREFIX.endswith("/"):
    R2_META_PREFIX += "/"

# Twitch clip thumbnails: ...-preview-480x272.jpg  -> mp4 real: ... .mp4
CLIP_PREVIEW_RE = re.compile(r"-preview-\d+x\d+\.jpg($|\?)", re.IGNORECASE)


# =========================
# R2
# =========================

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or "").rstrip("/")
R2_INBOX_PREFIX = (env_nonempty("R2_INBOX_PREFIX", "ugc/inbox") or "ugc/inbox").strip().strip("/")


# =========================
# Helpers
# =========================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().strftime("%Y-%m-%d")


def iso_now_full() -> str:
    return now_utc().isoformat()


def short_hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:10]


def safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9\-_]+", "_", s)[:120]
    return s.strip("_") or "clip"


def safe_json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def is_probably_mp4_bytes(data: bytes) -> bool:
    if not data or len(data) < 32:
        return False
    return b"ftyp" in data[:256]


def content_type_looks_video(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return ("video/" in ct) or ("mp4" in ct) or ("octet-stream" in ct)


# =========================
# State
# =========================

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


def load_state() -> Dict[str, Any]:
    st = s3_get_json(STATE_KEY)
    if not st:
        st = {"processed_clip_ids": [], "last_run_at": None}
    st.setdefault("processed_clip_ids", [])
    st.setdefault("last_run_at", None)
    return st


def save_state(st: Dict[str, Any]):
    st["last_run_at"] = iso_now_full()
    s3_put_json(STATE_KEY, st)


# =========================
# Twitch API
# =========================

def twitch_app_token() -> str:
    if not (TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET):
        raise RuntimeError("Faltan env Twitch: TWITCH_CLIENT_ID y TWITCH_CLIENT_SECRET")

    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": TWITCH_CLIENT_ID,
        "client_secret": TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    r = requests.post(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    tok = j.get("access_token")
    if not tok:
        raise RuntimeError(f"Twitch no devolvió access_token: {j}")
    return tok


def twitch_headers(token: str) -> Dict[str, str]:
    return {
        "Client-ID": TWITCH_CLIENT_ID or "",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }


def twitch_get_game_id(token: str, game_name: str) -> Optional[str]:
    url = "https://api.twitch.tv/helix/games"
    r = requests.get(url, headers=twitch_headers(token), params={"name": game_name}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = (r.json() or {}).get("data") or []
    if not data:
        return None
    return data[0].get("id")


def twitch_get_clips(token: str, game_id: str, first: int, started_at_iso: str) -> List[Dict[str, Any]]:
    url = "https://api.twitch.tv/helix/clips"
    params = {
        "game_id": game_id,
        "first": int(max(1, min(100, first))),
        "started_at": started_at_iso,
    }
    r = requests.get(url, headers=twitch_headers(token), params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return (r.json() or {}).get("data") or []


def clip_mp4_from_thumbnail(thumbnail_url: str) -> Optional[str]:
    if not thumbnail_url:
        return None
    u = thumbnail_url.strip()
    if not u.startswith("http"):
        return None
    if CLIP_PREVIEW_RE.search(u):
        u2 = CLIP_PREVIEW_RE.sub(".mp4", u)
        return u2.split("?")[0]
    if u.lower().endswith(".mp4"):
        return u
    return None


# =========================
# Viral scoring
# =========================

def twitch_score_clip(clip: Dict[str, Any], game_name: str) -> float:
    views = int(clip.get("view_count") or 0)
    title = (clip.get("title") or "").lower()

    created_at = clip.get("created_at") or ""
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_hours = max(0.1, (now_utc() - dt).total_seconds() / 3600.0)
    except Exception:
        age_hours = 72.0

    bonus = 0

    hot_words = [
        "clutch", "ace", "insane", "rage", "crazy", "1v", "1vs", "penta", "record",
        "headshot", "outplay", "faint", "fails", "final", "retake", "noscope"
    ]
    for w in hot_words:
        if w in title:
            bonus += 180

    if game_name.lower() in ("valorant", "counter-strike", "league of legends", "apex legends"):
        bonus += 100

    recency = 1500 / (1 + age_hours)

    return (views * 0.8) + recency + bonus


# =========================
# Download + validation
# =========================

def download_video_candidate(url: str) -> Tuple[bytes, str, int]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, allow_redirects=True)
    r.raise_for_status()

    content_type = (r.headers.get("Content-Type") or "").strip()
    try:
        content_length = int(r.headers.get("Content-Length") or 0)
    except Exception:
        content_length = 0

    return r.content, content_type, content_length


def validate_downloaded_clip(data: bytes, content_type: str, declared_length: int) -> Tuple[bool, str]:
    size = len(data)

    if size < TWITCH_MIN_VIDEO_BYTES:
        return False, f"too_small:{size}"

    if STRICT_VIDEO_CONTENT_TYPE and not content_type_looks_video(content_type):
        return False, f"bad_content_type:{content_type}"

    if not is_probably_mp4_bytes(data):
        return False, "missing_ftyp"

    if declared_length > 0 and abs(declared_length - size) > max(1024, int(declared_length * 0.20)):
        return False, f"length_mismatch:{declared_length}!={size}"

    return True, "ok"


# =========================
# Main
# =========================

def run_mode_e() -> None:
    print("===== MODE E (TWITCH VIRAL) -> R2 INBOX =====")
    print("TWITCH_MIN_VIDEO_BYTES:", TWITCH_MIN_VIDEO_BYTES)
    print("STRICT_VIDEO_CONTENT_TYPE:", STRICT_VIDEO_CONTENT_TYPE)

    state = load_state()
    processed_ids = set(state.get("processed_clip_ids", []))

    token = twitch_app_token()

    started_at = (now_utc() - timedelta(days=TWITCH_PERIOD_DAYS)).isoformat().replace("+00:00", "Z")

    game_names = [x.strip() for x in (TWITCH_GAMES or "").split(",") if x.strip()]
    if not game_names:
        raise RuntimeError("TWITCH_GAMES vacío. Pon algo tipo: 'VALORANT,League of Legends,Counter-Strike'")

    all_candidates: List[Tuple[float, Dict[str, Any]]] = []

    for g in game_names:
        print("Exploring:", g)
        gid = twitch_get_game_id(token, g)
        if not gid:
            print(" - No game_id encontrado, skip")
            continue

        clips = twitch_get_clips(token, gid, first=TWITCH_CLIPS_PER_GAME, started_at_iso=started_at)
        for c in clips:
            clip_id = c.get("id")
            if not clip_id or clip_id in processed_ids:
                continue

            views = int(c.get("view_count") or 0)
            thumb = c.get("thumbnail_url") or ""
            mp4 = clip_mp4_from_thumbnail(thumb)

            if not mp4:
                continue
            if views < TWITCH_MIN_VIEWS:
                continue

            score = twitch_score_clip(c, g)

            all_candidates.append((
                score,
                {
                    "game": g,
                    "id": clip_id,
                    "title": c.get("title") or "",
                    "creator": c.get("creator_name") or "",
                    "views": views,
                    "mp4": mp4,
                    "url": c.get("url") or "",
                    "created_at": c.get("created_at") or "",
                    "broadcaster_name": c.get("broadcaster_name") or "",
                    "thumbnail_url": thumb,
                    "score": score,
                }
            ))

    if not all_candidates:
        print("No hay clips candidatos con los filtros actuales.")
        return

    all_candidates.sort(key=lambda x: x[0], reverse=True)

    # mezcla un poco para no ser demasiado robótico, pero sin perder calidad
    top_pool = all_candidates[:max(TWITCH_PICK_TOTAL * 3, TWITCH_PICK_TOTAL)]
    random.shuffle(top_pool)
    top_pool.sort(key=lambda x: x[0], reverse=True)
    picked = top_pool[:max(1, TWITCH_PICK_TOTAL)]

    print(f"Seleccionados {len(picked)} clips para subir a inbox.")

    uploaded_count = 0

    for score, c in picked:
        mp4_url = c["mp4"]
        views = c.get("views", 0)
        title = c.get("title", "")
        clip_id = c.get("id", "unknown")

        print("Downloading:", mp4_url)

        try:
            data, content_type, declared_length = download_video_candidate(mp4_url)

            print("Downloaded bytes:", len(data))
            print("Content-Type:", content_type or "(none)")
            print("Content-Length header:", declared_length)

            ok, reason = validate_downloaded_clip(data, content_type, declared_length)
            if not ok:
                print("Clip descartado:", reason)
                continue

        except Exception as e:
            print("Download error:", str(e))
            continue

        h = short_hash_bytes(data)
        key = f"{R2_INBOX_PREFIX}/{iso_now()}__twitch__{clip_id}__{safe_slug(title)}__{h}.mp4"

        print("Uploading to R2 inbox:", key)
        upload_bytes_to_r2(key, data, "video/mp4")

        meta = {
            "source": "twitch",
            "clip_id": clip_id,
            "title": title,
            "creator": c.get("creator"),
            "views": views,
            "url": c.get("url"),
            "video_url": mp4_url,
            "game_name": c.get("game"),
            "broadcaster_name": c.get("broadcaster_name"),
            "thumbnail_url": c.get("thumbnail_url"),
            "created_at": c.get("created_at"),
            "score": score,
            "queued_at": iso_now_full(),
            "r2_key": key,
        }

        meta_base = os.path.basename(key).rsplit(".", 1)[0] + ".json"
        meta_key = f"{R2_META_PREFIX}{meta_base}"
        s3_put_json(meta_key, meta)

        processed_ids.add(clip_id)
        uploaded_count += 1

        print(f"OK: {title[:80]} | views: {views} | score: {round(score, 2)}")
        print("Saved meta:", meta_key)

    state["processed_clip_ids"] = list(processed_ids)[-2000:]
    save_state(state)

    print("===== MODE E DONE =====")
    print("Uploaded valid clips:", uploaded_count)


if __name__ == "__main__":
    run_mode_e()
