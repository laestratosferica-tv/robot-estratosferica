# ugc_mode_f.py
import os
import re
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple

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
# CONFIG
# -------------------------

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

# R2
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")

UGC_INBOX_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox/") or "ugc/inbox/").strip()
if not UGC_INBOX_PREFIX.endswith("/"):
    UGC_INBOX_PREFIX += "/"

STATE_KEY = env_nonempty("VIRAL_STATE_KEY", "ugc/state/viral_state.json")

IDEAS_PREFIX = (env_nonempty("VIRAL_IDEAS_PREFIX", "ugc/ideas/") or "ugc/ideas/").strip()
if not IDEAS_PREFIX.endswith("/"):
    IDEAS_PREFIX += "/"

UGC_META_PREFIX = (env_nonempty("UGC_META_PREFIX", "ugc/meta/") or "ugc/meta/").strip()
if not UGC_META_PREFIX.endswith("/"):
    UGC_META_PREFIX += "/"

# Twitch
TWITCH_CLIENT_ID = env_nonempty("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = env_nonempty("TWITCH_CLIENT_SECRET")
TWITCH_USER_ACCESS_TOKEN = env_nonempty("TWITCH_USER_ACCESS_TOKEN")
TWITCH_BROADCASTER_ID = env_nonempty("TWITCH_BROADCASTER_ID")
TWITCH_USE_OFFICIAL_DOWNLOAD = env_bool("TWITCH_USE_OFFICIAL_DOWNLOAD", False)

TWITCH_MAX_CLIPS_PER_RUN = env_int("TWITCH_MAX_CLIPS_PER_RUN", 3)
TWITCH_LOOKBACK_HOURS = env_int("TWITCH_LOOKBACK_HOURS", 24)
TWITCH_MIN_VIEWS = env_int("TWITCH_MIN_VIEWS", 2500)
TWITCH_TOP_GAMES = env_int("TWITCH_TOP_GAMES", 10)

# validación de archivos descargados
TWITCH_MIN_VIDEO_BYTES = env_int("TWITCH_MIN_VIDEO_BYTES", 200_000)
STRICT_VIDEO_CONTENT_TYPE = env_bool("STRICT_VIDEO_CONTENT_TYPE", False)

# Reddit
REDDIT_ENABLED = env_bool("REDDIT_ENABLED", True)
REDDIT_SUBS = [
    s.strip() for s in (
        env_nonempty(
            "REDDIT_SUBS",
            "esports,leagueoflegends,GlobalOffensive,VALORANT,Competitiveoverwatch,gaming"
        ) or ""
    ).split(",")
    if s.strip()
]
REDDIT_MIN_SCORE = env_int("REDDIT_MIN_SCORE", 400)

# OpenAI optional for idea polishing
OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")

DRY_RUN = env_bool("DRY_RUN", False)

USER_AGENT = env_nonempty(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
) or "Mozilla/5.0"

# fallback viejo thumbnail -> mp4
CLIP_PREVIEW_RE = re.compile(r"-preview-\d+x\d+\.jpg($|\?)", re.IGNORECASE)


# -------------------------
# Generic helpers
# -------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat()


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def sanitize_filename(name: str) -> str:
    name = (name or "clip").strip()
    name = re.sub(r"[^\w\-. ]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:150] or "clip"


def safe_json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def is_probably_mp4_bytes(data: bytes) -> bool:
    if not data or len(data) < 32:
        return False
    head = data[:256]
    return b"ftyp" in head


def content_type_looks_video(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return ("video/" in ct) or ("mp4" in ct) or ("octet-stream" in ct)


# -------------------------
# R2 helpers
# -------------------------

def r2_client():
    if not (BUCKET_NAME and R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise RuntimeError("Faltan credenciales R2/S3")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def s3_put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream"):
    r2_client().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_get_json(key: str):
    try:
        obj = r2_client().get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def s3_put_json(key: str, payload: Dict[str, Any]):
    s3_put_bytes(key, safe_json_dumps(payload), "application/json")


# -------------------------
# State
# -------------------------

def load_state() -> Dict[str, Any]:
    st = s3_get_json(STATE_KEY)
    if not st:
        st = {
            "processed_twitch_clip_ids": [],
            "processed_reddit_posts": [],
            "last_run_at": None,
        }
    st.setdefault("processed_twitch_clip_ids", [])
    st.setdefault("processed_reddit_posts", [])
    st.setdefault("last_run_at", None)
    return st


def save_state(st: Dict[str, Any]):
    st["last_run_at"] = iso_now()
    s3_put_json(STATE_KEY, st)


# -------------------------
# Twitch
# -------------------------

def twitch_get_app_token() -> str:
    if not (TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET):
        raise RuntimeError("Faltan TWITCH_CLIENT_ID y TWITCH_CLIENT_SECRET")
    r = requests.post(
        "https://id.twitch.tv/oauth2/token",
        params={
            "client_id": TWITCH_CLIENT_ID,
            "client_secret": TWITCH_CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def twitch_headers(token: str) -> Dict[str, str]:
    return {
        "Client-Id": TWITCH_CLIENT_ID or "",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }


def twitch_get_top_games(token: str) -> List[Dict[str, Any]]:
    r = requests.get(
        "https://api.twitch.tv/helix/games/top",
        headers=twitch_headers(token),
        params={"first": TWITCH_TOP_GAMES},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("data", [])


def twitch_get_clips_for_game(token: str, game_id: str) -> List[Dict[str, Any]]:
    start = now_utc() - timedelta(hours=TWITCH_LOOKBACK_HOURS)
    r = requests.get(
        "https://api.twitch.tv/helix/clips",
        headers=twitch_headers(token),
        params={
            "game_id": game_id,
            "started_at": start.isoformat().replace("+00:00", "Z"),
            "first": 20,
        },
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("data", [])


def twitch_score_clip(clip: Dict[str, Any]) -> float:
    views = int(clip.get("view_count", 0))
    created_at = clip.get("created_at")
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_hours = max(0.1, (now_utc() - dt).total_seconds() / 3600.0)
    except Exception:
        age_hours = 48.0

    title = (clip.get("title") or "").lower()
    bonus = 0
    hot_words = ["clutch", "ace", "insane", "rage", "crazy", "1v", "penta", "top 1", "record"]
    for w in hot_words:
        if w in title:
            bonus += 200

    return (views * 0.75) + (1200 / (1 + age_hours)) + bonus


def twitch_clip_mp4_url(clip: Dict[str, Any]) -> Optional[str]:
    thumb = (clip.get("thumbnail_url") or "").strip()
    if not thumb:
        return None
    if CLIP_PREVIEW_RE.search(thumb):
        return CLIP_PREVIEW_RE.sub(".mp4", thumb).split("?")[0]
    if thumb.lower().endswith(".mp4"):
        return thumb
    return None


def twitch_get_clip_download_url(clip_id: str) -> Optional[str]:
    if not TWITCH_USER_ACCESS_TOKEN:
        print("OFFICIAL DOWNLOAD skip: falta TWITCH_USER_ACCESS_TOKEN")
        return None

    if not TWITCH_BROADCASTER_ID:
        print("OFFICIAL DOWNLOAD skip: falta TWITCH_BROADCASTER_ID")
        return None

    if not clip_id:
        return None

    url = "https://api.twitch.tv/helix/clips/downloads"
    params = {
        "editor_id": TWITCH_BROADCASTER_ID,
        "broadcaster_id": TWITCH_BROADCASTER_ID,
        "clip_id": clip_id,
    }

    r = requests.get(
        url,
        headers=twitch_headers(TWITCH_USER_ACCESS_TOKEN),
        params=params,
        timeout=HTTP_TIMEOUT,
    )

    if r.status_code == 401:
        print("OFFICIAL DOWNLOAD 401: token inválido o expirado")
        return None

    if r.status_code == 403:
        print("OFFICIAL DOWNLOAD 403: sin permisos para descargar clip", clip_id)
        return None

    if r.status_code == 400:
        print("OFFICIAL DOWNLOAD 400:", r.text[:300])
        return None

    r.raise_for_status()

    data = (r.json() or {}).get("data") or []
    if not data:
        print("OFFICIAL DOWNLOAD: Twitch no devolvió URL para clip", clip_id)
        return None

    item = data[0] or {}
    return item.get("landscape_download_url") or item.get("portrait_download_url")


def download_video_candidate(url: str) -> Tuple[bytes, str, int]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, allow_redirects=True)
    r.raise_for_status()

    content_type = (r.headers.get("Content-Type") or "").strip()
    content_length = 0
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


# -------------------------
# Reddit trends / ideas
# -------------------------

def reddit_fetch_hot(sub: str) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/r/{sub}/hot.json"
    r = requests.get(
        url,
        headers={"User-Agent": "robot-gamer-trend-bot/1.0"},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    posts = r.json().get("data", {}).get("children", [])
    out = []
    for p in posts:
        d = p.get("data", {})
        if int(d.get("score", 0)) >= REDDIT_MIN_SCORE:
            out.append({
                "id": d.get("id"),
                "subreddit": sub,
                "title": d.get("title"),
                "score": d.get("score", 0),
                "url": d.get("url"),
                "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
            })
    return out


def openai_text(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
    }
    r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("output_text"):
        return j["output_text"].strip()

    texts = []
    for item in j.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                texts.append(part["text"])
    return "\n".join(texts).strip()


def build_reddit_idea(post: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Eres editor viral gamer LATAM.
Convierte esta tendencia de Reddit en una idea de reel gaming polémico/comentable.
Devuelve SOLO JSON válido con:
{{
  "hook": "...",
  "angle": "...",
  "caption_seed": "...",
  "why_it_hits": "..."
}}

Tendencia:
Subreddit: {post['subreddit']}
Título: {post['title']}
Score: {post['score']}
"""
    try:
        raw = openai_text(prompt)
        return json.loads(raw)
    except Exception:
        return {
            "hook": post["title"][:90],
            "angle": "tema caliente gamer con potencial de debate",
            "caption_seed": "Esto va a dividir a la comunidad.",
            "why_it_hits": "alta conversación en Reddit gaming",
        }


# -------------------------
# Main Mode F
# -------------------------

def run_mode_f():
    print("===== MODE F VIRAL GAMER BRAIN =====")
    print("DRY_RUN:", DRY_RUN)
    print("UGC_INBOX_PREFIX:", UGC_INBOX_PREFIX)
    print("UGC_META_PREFIX:", UGC_META_PREFIX)
    print("IDEAS_PREFIX:", IDEAS_PREFIX)
    print("TWITCH_USE_OFFICIAL_DOWNLOAD:", TWITCH_USE_OFFICIAL_DOWNLOAD)
    print("TWITCH_MIN_VIDEO_BYTES:", TWITCH_MIN_VIDEO_BYTES)
    print("STRICT_VIDEO_CONTENT_TYPE:", STRICT_VIDEO_CONTENT_TYPE)

    if TWITCH_USE_OFFICIAL_DOWNLOAD:
        if not TWITCH_USER_ACCESS_TOKEN:
            raise RuntimeError("TWITCH_USE_OFFICIAL_DOWNLOAD=true pero falta TWITCH_USER_ACCESS_TOKEN")
        if not TWITCH_BROADCASTER_ID:
            raise RuntimeError("TWITCH_USE_OFFICIAL_DOWNLOAD=true pero falta TWITCH_BROADCASTER_ID")

    state = load_state()

    # -------- Twitch scanning --------
    twitch_candidates: List[Tuple[float, Dict[str, Any], str]] = []

    if TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET:
        try:
            token = twitch_get_app_token()
            games = twitch_get_top_games(token)

            for game in games[:TWITCH_TOP_GAMES]:
                game_name = game.get("name", "unknown")
                game_id = game.get("id")
                print("Exploring Twitch game:", game_name)

                clips = twitch_get_clips_for_game(token, game_id)

                for clip in clips:
                    clip_id = clip.get("id")
                    if not clip_id or clip_id in state["processed_twitch_clip_ids"]:
                        continue

                    views = int(clip.get("view_count", 0))
                    if views < TWITCH_MIN_VIEWS:
                        continue

                    score = twitch_score_clip(clip)
                    twitch_candidates.append((score, clip, game_name))
        except Exception as e:
            print("Twitch scan falló (no rompe):", str(e))
    else:
        print("Twitch creds faltan. Saltando Twitch.")

    twitch_candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = twitch_candidates[:TWITCH_MAX_CLIPS_PER_RUN]

    print(f"Twitch clips elegidos: {len(chosen)}")

    for score, clip, game_name in chosen:
        clip_id = clip["id"]
        title = clip.get("title") or f"clip_{clip_id}"

        key_name = sanitize_filename(title) + ".mp4"
        timestamp = now_utc().strftime("%Y-%m-%d__%H%M%S")
        r2_key = f"{UGC_INBOX_PREFIX}{timestamp}__twitch__{clip_id}__{key_name}"

        video_url = None
        download_method = None

        try:
            if TWITCH_USE_OFFICIAL_DOWNLOAD:
                video_url = twitch_get_clip_download_url(clip_id)
                download_method = "official_api"
                if not video_url:
                    print("Clip descartado: sin official download url:", clip_id)
                    continue
            else:
                video_url = twitch_clip_mp4_url(clip)
                download_method = "thumbnail_fallback"
                if not video_url:
                    print("Clip descartado: no se pudo derivar mp4 desde thumbnail:", clip_id)
                    continue

            meta = {
                "source": "twitch",
                "clip_id": clip_id,
                "title": clip.get("title"),
                "view_count": clip.get("view_count"),
                "created_at": clip.get("created_at"),
                "url": clip.get("url"),
                "video_url": video_url,
                "download_method": download_method,
                "broadcaster_name": clip.get("broadcaster_name"),
                "game_name": game_name,
                "score": score,
                "queued_at": iso_now(),
                "r2_key": r2_key,
            }

            print("Selected Twitch clip:", clip.get("title"), "| score:", round(score, 2), "| views:", clip.get("view_count"))
            print("Download method:", download_method)
            print("Video URL:", video_url)

            if DRY_RUN:
                print("[DRY_RUN] Subiría clip a:", r2_key)
                continue

            data, content_type, declared_length = download_video_candidate(video_url)

            print("Downloaded bytes:", len(data))
            print("Content-Type:", content_type or "(none)")
            print("Content-Length header:", declared_length)

            ok, reason = validate_downloaded_clip(data, content_type, declared_length)
            if not ok:
                print("Clip descartado:", reason)
                continue

            s3_put_bytes(r2_key, data, "video/mp4")

            meta_base = os.path.basename(r2_key).rsplit(".", 1)[0] + ".json"
            meta_key = f"{UGC_META_PREFIX}{meta_base}"
            s3_put_json(meta_key, meta)

            state["processed_twitch_clip_ids"].append(clip_id)
            state["processed_twitch_clip_ids"] = state["processed_twitch_clip_ids"][-1000:]

            print("Queued to inbox:", r2_key)
            print("Saved meta:", meta_key)

        except Exception as e:
            print("No se pudo subir clip Twitch:", str(e))

    # -------- Reddit gamer trends --------
    if REDDIT_ENABLED and REDDIT_SUBS:
        for sub in REDDIT_SUBS:
            try:
                posts = reddit_fetch_hot(sub)
                for post in posts[:3]:
                    pid = post["id"]
                    state_key = f"{sub}:{pid}"
                    if state_key in state["processed_reddit_posts"]:
                        continue

                    idea = build_reddit_idea(post)
                    payload = {
                        "source": "reddit",
                        "post": post,
                        "idea": idea,
                        "generated_at": iso_now(),
                    }

                    idea_key = f"{IDEAS_PREFIX}{now_utc().strftime('%Y-%m-%d__%H%M%S')}__reddit__{sub}__{pid}.json"

                    if DRY_RUN:
                        print("[DRY_RUN] Guardaría idea:", idea_key)
                    else:
                        s3_put_json(idea_key, payload)
                        state["processed_reddit_posts"].append(state_key)
                        state["processed_reddit_posts"] = state["processed_reddit_posts"][-1000:]
                        print("Saved gamer idea:", idea_key)
            except Exception as e:
                print(f"Reddit scan falló en r/{sub}:", str(e))

    save_state(state)
    print("===== MODE F DONE =====")


if __name__ == "__main__":
    run_mode_f()
