# ===== ugc_mode_b_feedback.py =====

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

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

STATE_KEY = env_nonempty("B_STATE_KEY", "ugc/state/mode_b_state.json")
MEMORY_KEY = env_nonempty("B_EDITORIAL_MEMORY_KEY", "ugc/state/editorial_memory.json")
SUMMARY_KEY = env_nonempty("B_EDITORIAL_MEMORY_SUMMARY_KEY", "ugc/state/editorial_memory_summary.json")

LOOKBACK_DAYS = env_int("B_EDITORIAL_LOOKBACK_DAYS", 7)


def now_utc():
    return datetime.now(timezone.utc)


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


def load_json(key):
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return {}


def save_json(key, data):
    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def parse_iso_datetime(value):
    if not value:
        return None

    try:
        v = str(value).strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def recent_items(history):
    cutoff = now_utc() - timedelta(days=LOOKBACK_DAYS)
    out = []

    for item in history:
        ts = item.get("published_at")
        dt = parse_iso_datetime(ts)
        if not dt:
            continue

        if dt >= cutoff:
            out.append(item)

    return out


def extract_words(text):
    words = []
    for w in str(text or "").lower().split():
        w = w.strip(".,!?;:()[]{}\"'`“”‘’#🔥-_/\\|")
        if len(w) < 4:
            continue
        if w.isdigit():
            continue
        words.append(w)
    return words


def build_editorial_memory(items):
    return {
        "version": "v2_r2",
        "generated_at": now_utc().isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "items": items,
    }


def build_summary(items):
    by_game = defaultdict(list)

    for it in items:
        game = (it.get("game_name") or "generic").strip().lower()
        by_game[game].append(it)

    summary_games = {}

    for game, rows in by_game.items():
        captions = [r.get("caption_final", "") for r in rows if r.get("caption_final")]
        questions = [c for c in captions if "¿" in c or "?" in c]

        freq = defaultdict(int)
        for caption in captions:
            for w in extract_words(caption):
                freq[w] += 1

        top_words = [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:12]]

        patterns_hint = [
            "usar conflicto",
            "usar pregunta final",
            "mencionar juego explícitamente",
        ]

        if len(questions) >= max(1, len(captions) // 2):
            patterns_hint.append("pregunta polarizante funciona")

        if any("skill" in c.lower() for c in captions):
            patterns_hint.append("skill vs suerte funciona")

        if any("inflado" in c.lower() for c in captions):
            patterns_hint.append("inflado vs real funciona")

        if any("lobby" in c.lower() for c in captions):
            patterns_hint.append("lobby / rival flojo funciona")

        if any("clutch" in c.lower() for c in captions):
            patterns_hint.append("clutch funciona")

        if any("regalo" in c.lower() or "regalado" in c.lower() for c in captions):
            patterns_hint.append("skill vs regalo funciona")

        summary_games[game] = {
            "recent_posts": len(rows),
            "top_words": top_words,
            "patterns_hint": patterns_hint[:8],
        }

    return {
        "version": "v2_r2",
        "generated_at": now_utc().isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "games": summary_games,
    }


def run():
    print("===== MODE B FEEDBACK START =====")
    print("STATE_KEY:", STATE_KEY)
    print("MEMORY_KEY:", MEMORY_KEY)
    print("SUMMARY_KEY:", SUMMARY_KEY)
    print("LOOKBACK_DAYS:", LOOKBACK_DAYS)

    state = load_json(STATE_KEY)
    history = state.get("history", [])

    if not history:
        print("No history found in mode_b_state.json")
        return

    recents = recent_items(history)
    if not recents:
        print("No recent items in lookback window.")
        return

    memory = build_editorial_memory(recents)
    summary = build_summary(recents)

    save_json(MEMORY_KEY, memory)
    save_json(SUMMARY_KEY, summary)

    print(f"Editorial memory updated. Items: {len(recents)}")
    print(f"Saved to R2: {MEMORY_KEY}")
    print(f"Saved to R2: {SUMMARY_KEY}")
    print("===== MODE B FEEDBACK DONE =====")


if __name__ == "__main__":
    run()
