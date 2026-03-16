# ===== ugc_mode_b_feedback.py =====

import json
import os
import re
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


STOPWORDS = {
    "esto", "esta", "está", "para", "pero", "porque", "como", "donde", "desde",
    "entre", "sobre", "hacia", "hasta", "tambien", "también", "solo", "sólo",
    "aqui", "aquí", "alla", "allá", "muy", "mas", "más", "menos", "casi",
    "puro", "pura", "total", "real", "brutal", "mucho", "mucha", "mejor",
    "peor", "igual", "siempre", "nunca", "nadie", "todos", "todas", "este",
    "esta", "estos", "estas", "ese", "esa", "esos", "esas", "una", "uno",
    "unos", "unas", "del", "las", "los", "con", "sin", "por", "que", "qué",
    "quien", "quién", "como", "cómo", "fue", "era", "son", "ser", "hay",
    "the", "and", "for", "with", "from", "this", "that", "your", "just",
}

BANNED_WORDS = {
    "gaminglatam",
    "esportslatam",
    "reelsgaming",
    "valorantlatam",
    "warzonelatam",
    "cs2latam",
    "f1esports",
    "callofduty",
    "counterstrike",
    "apexlatam",
    "fortnitelatam",
    "minecraftlatam",
    "gltalam",
    "shorts",
    "reels",
    "tiktok",
    "instagram",
    "facebook",
    "youtube",
    "latam",
}

BANNED_PREFIXES = (
    "http",
    "www",
)

RECENT_PHRASE_KEYS = [
    "en momento de manos",
    "la mayoría la vende aquí",
    "esto no es highlight, es castigo",
    "momento de manos y sangre fría",
    "la mayoría aquí la vende",
    "la mayoría se apaga aquí",
]


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


def clean_line(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def split_caption_lines(text):
    return [clean_line(x) for x in str(text or "").splitlines() if clean_line(x)]


def is_hashtag_line(line):
    if not line:
        return False
    tokens = line.split()
    if not tokens:
        return False
    tagged = 0
    for t in tokens:
        if t.startswith("#"):
            tagged += 1
    return tagged >= max(1, len(tokens) - 1)


def strip_hashtags_from_caption(caption):
    lines = split_caption_lines(caption)
    kept = []

    for line in lines:
        if is_hashtag_line(line):
            continue
        kept.append(line)

    return "\n".join(kept).strip()


def normalize_word(word):
    w = str(word or "").lower().strip()
    w = w.strip(".,!?;:()[]{}\"'`“”‘’#🔥-_/\\|")
    return w


def should_keep_word(word):
    w = normalize_word(word)
    if not w:
        return False
    if len(w) < 4:
        return False
    if w.isdigit():
        return False
    if w in STOPWORDS:
        return False
    if w in BANNED_WORDS:
        return False
    if any(w.startswith(p) for p in BANNED_PREFIXES):
        return False
    if "#" in w:
        return False
    return True


def extract_words(text):
    words = []
    for raw in str(text or "").lower().split():
        w = normalize_word(raw)
        if should_keep_word(w):
            words.append(w)
    return words


def detect_patterns(captions):
    hints = [
        "usar conflicto",
        "usar pregunta final",
        "mencionar juego explícitamente",
    ]

    joined = "\n".join(captions).lower()

    if any("¿" in c or "?" in c for c in captions):
        hints.append("pregunta polarizante funciona")

    if any(x in joined for x in ["skill", "suerte", "regalo", "regalado"]):
        hints.append("skill vs suerte funciona")

    if any(x in joined for x in ["inflado", "humo", "vendiendo de más", "vendiendo de mas"]):
        hints.append("inflado vs real funciona")

    if any(x in joined for x in ["lobby", "rival", "dormido", "regaló", "regalo"]):
        hints.append("rival flojo / regalo funciona")

    if any(x in joined for x in ["clutch", "sentencia", "borra", "castiga"]):
        hints.append("clutch / castigo funciona")

    if any(x in joined for x in ["caos", "milagro", "bendecido"]):
        hints.append("caos vs mérito funciona")

    return hints[:8]


def build_recent_phrase_counts(captions):
    counts = {}
    joined = "\n".join(captions).lower()

    for phrase in RECENT_PHRASE_KEYS:
        counts[phrase] = joined.count(phrase)

    return counts


def build_editorial_memory(items):
    cleaned_items = []

    for item in items:
        cloned = dict(item)
        cloned["caption_editorial_body"] = strip_hashtags_from_caption(item.get("caption_final", ""))
        cleaned_items.append(cloned)

    return {
        "version": "v3_r2_clean",
        "generated_at": now_utc().isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "items": cleaned_items,
    }


def build_summary(items):
    by_game = defaultdict(list)

    for it in items:
        game = (it.get("game_name") or "generic").strip().lower()
        by_game[game].append(it)

    summary_games = {}

    for game, rows in by_game.items():
        raw_captions = [r.get("caption_final", "") for r in rows if r.get("caption_final")]
        captions = [strip_hashtags_from_caption(c) for c in raw_captions if strip_hashtags_from_caption(c)]

        freq = defaultdict(int)
        for caption in captions:
            for w in extract_words(caption):
                freq[w] += 1

        top_words = [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:12]]
        patterns_hint = detect_patterns(captions)
        recent_phrase_counts = build_recent_phrase_counts(captions)

        summary_games[game] = {
            "recent_posts": len(rows),
            "top_words": top_words,
            "patterns_hint": patterns_hint,
            "recent_phrase_counts": recent_phrase_counts,
        }

    return {
        "version": "v3_r2_clean",
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
