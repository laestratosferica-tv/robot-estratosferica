# ===== ugc_mode_b_feedback.py =====

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta


STATE_PATH = "ugc/state/mode_b_state.json"
MEMORY_PATH = "ugc/state/editorial_memory.json"
SUMMARY_PATH = "ugc/state/editorial_memory_summary.json"

LOOKBACK_DAYS = 7


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_iso_datetime(value):
    if not value:
        return None

    try:
        v = str(value).strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        return datetime.fromisoformat(v)
    except Exception:
        return None


def recent_items(history):
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    out = []

    for item in history:
        ts = item.get("published_at")
        dt = parse_iso_datetime(ts)
        if not dt:
            continue

        if dt.tzinfo is not None:
            dt_naive = dt.astimezone().replace(tzinfo=None)
        else:
            dt_naive = dt

        if dt_naive >= cutoff:
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
        "version": "v1",
        "generated_at": datetime.utcnow().isoformat(),
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

        summary_games[game] = {
            "recent_posts": len(rows),
            "top_words": top_words,
            "patterns_hint": patterns_hint[:8],
        }

    return {
        "version": "v1",
        "generated_at": datetime.utcnow().isoformat(),
        "games": summary_games,
    }


def run():
    state = load_json(STATE_PATH)
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

    save_json(MEMORY_PATH, memory)
    save_json(SUMMARY_PATH, summary)

    print(f"Editorial memory updated. Items: {len(recents)}")
    print(f"Saved: {MEMORY_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    run()
