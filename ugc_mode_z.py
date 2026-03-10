# ugc_mode_z.py
import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

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
# Config
# =========================

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

REDDIT_SUBS = env_nonempty(
    "REDDIT_GAMER_SUBS",
    "ValorantCompetitive,GlobalOffensive,leagueoflegends,FortniteCompetitive,CODWarzone,apexlegends,Minecraft,fifacareers,simracing,granturismo,F1Game"
)

REDDIT_MAX_POSTS_PER_SUB = env_int("REDDIT_MAX_POSTS_PER_SUB", 10)
REDDIT_MIN_SCORE = env_int("REDDIT_MIN_SCORE", 100)
REDDIT_MIN_COMMENTS = env_int("REDDIT_MIN_COMMENTS", 15)

STATE_KEY = env_nonempty("REDDIT_MODE_Z_STATE_KEY", "ugc/state/mode_z_state.json")
IDEAS_PREFIX = (env_nonempty("VIRAL_IDEAS_PREFIX", "ugc/ideas/") or "ugc/ideas/").strip()
if not IDEAS_PREFIX.endswith("/"):
    IDEAS_PREFIX += "/"

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")

REDDIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 EstratosfericaBot/1.0",
    "Accept": "application/json",
}


# =========================
# Generic helpers
# =========================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now_full() -> str:
    return now_utc().isoformat()


def safe_json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


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


def s3_put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream"):
    r2_client().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


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
        st = {"processed_posts": [], "last_run_at": None}
    st.setdefault("processed_posts", [])
    st.setdefault("last_run_at", None)
    return st


def save_state(st: Dict[str, Any]):
    st["last_run_at"] = iso_now_full()
    s3_put_json(STATE_KEY, st)


# =========================
# Reddit
# =========================

def reddit_fetch_hot(sub: str) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/r/{sub}/hot.json?limit={REDDIT_MAX_POSTS_PER_SUB}"

    r = requests.get(
        url,
        headers=REDDIT_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()

    payload = r.json() or {}
    children = payload.get("data", {}).get("children", [])

    items: List[Dict[str, Any]] = []

    for child in children:
        d = child.get("data", {}) or {}

        score = int(d.get("score") or 0)
        comments = int(d.get("num_comments") or 0)

        if score < REDDIT_MIN_SCORE:
            continue
        if comments < REDDIT_MIN_COMMENTS:
            continue

        items.append(
            {
                "id": d.get("id"),
                "subreddit": sub,
                "title": d.get("title") or "",
                "score": score,
                "num_comments": comments,
                "author": d.get("author") or "",
                "url": d.get("url") or "",
                "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
                "created_utc": d.get("created_utc"),
                "is_video": bool(d.get("is_video") or False),
                "domain": d.get("domain") or "",
            }
        )

    return items


def reddit_score(post: Dict[str, Any]) -> float:
    score = int(post.get("score") or 0)
    comments = int(post.get("num_comments") or 0)
    title = (post.get("title") or "").lower()
    subreddit = (post.get("subreddit") or "").lower()

    bonus = 0

    hot_words = [
        "clutch", "ace", "patch", "broken", "meta", "nerf", "buff",
        "leak", "insane", "goal", "career mode", "lap", "overtake",
        "ranked", "rage", "comeback", "best build"
    ]

    for w in hot_words:
        if w in title:
            bonus += 40

    if subreddit in {"valorantcompetitive", "globaloffensive", "leagueoflegends", "fortnitecompetitive"}:
        bonus += 30

    return (score * 1.0) + (comments * 2.0) + bonus


def build_idea_payload(post: Dict[str, Any]) -> Dict[str, Any]:
    title = post.get("title") or ""
    sub = post.get("subreddit") or ""

    return {
        "source": "reddit",
        "type": "trend_idea",
        "generated_at": iso_now_full(),
        "post": post,
        "idea": {
            "hook": f"¿Esto cambia el meta en {sub}?",
            "angle": f"Tema caliente en r/{sub}: {title[:120]}",
            "caption_seed": "Esto está prendiendo a la comunidad gamer. ¿Tú qué opinas?",
            "why_it_hits": "alto score + muchos comentarios + potencial de debate",
        },
    }


# =========================
# Main
# =========================

def run_mode_z():
    print("===== MODE Z (REDDIT GAMER RADAR) =====")
    print("REDDIT_GAMER_SUBS:", REDDIT_SUBS)
    print("REDDIT_MAX_POSTS_PER_SUB:", REDDIT_MAX_POSTS_PER_SUB)
    print("REDDIT_MIN_SCORE:", REDDIT_MIN_SCORE)
    print("REDDIT_MIN_COMMENTS:", REDDIT_MIN_COMMENTS)
    print("STATE_KEY:", STATE_KEY)

    state = load_state()
    processed = set(state.get("processed_posts", []))
    print("Processed reddit posts in state:", len(processed))

    subs = [x.strip() for x in (REDDIT_SUBS or "").split(",") if x.strip()]
    if not subs:
        raise RuntimeError("REDDIT_GAMER_SUBS vacío")

    all_candidates: List[Tuple[float, Dict[str, Any]]] = []

    for sub in subs:
        try:
            print("Exploring subreddit:", sub)
            posts = reddit_fetch_hot(sub)
            print(f"Found {len(posts)} posts in r/{sub}")

            for post in posts:
                pid = post.get("id")
                if not pid:
                    continue

                post_state_key = f"{sub}:{pid}"
                if post_state_key in processed:
                    print("SKIP already processed:", post_state_key)
                    continue

                score = reddit_score(post)
                print(
                    "CANDIDATE:",
                    post_state_key,
                    "| score:", round(score, 2),
                    "|", (post.get("title") or "")[:100],
                )
                all_candidates.append((score, post))

        except Exception as e:
            print(f"Reddit scan falló en r/{sub}:", str(e))

    if not all_candidates:
        print("No hay posts candidatos.")
        save_state(state)
        print("===== MODE Z DONE =====")
        return

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    picked = all_candidates[: min(10, len(all_candidates))]

    saved_count = 0

    for score, post in picked:
        sub = post.get("subreddit") or "unknown"
        pid = post.get("id") or "unknown"
        post_state_key = f"{sub}:{pid}"

        payload = build_idea_payload(post)
        file_hash = short_hash(post_state_key + (post.get("title") or ""))
        idea_key = f"{IDEAS_PREFIX}{now_utc().strftime('%Y-%m-%d__%H%M%S')}__reddit__{sub}__{pid}__{file_hash}.json"

        s3_put_json(idea_key, payload)
        processed.add(post_state_key)
        saved_count += 1

        print("Saved gamer idea:", idea_key)

    state["processed_posts"] = list(processed)[-5000:]
    save_state(state)

    print("===== MODE Z DONE =====")
    print("Saved ideas:", saved_count)


if __name__ == "__main__":
    run_mode_z()
