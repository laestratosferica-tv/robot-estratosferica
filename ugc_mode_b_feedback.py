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


def recent_items(history):
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    out = []

    for item in history:
        ts = item.get("published_at")
        if not ts:
            continue

        try:
            dt = datetime.fromisoformat(ts.replace("Z", ""))
        except:
            continue

        if dt >= cutoff:
            out.append(item)

    return out


def build_editorial_memory(items):
    memory = {
        "version": "v1",
        "generated_at": datetime.utcnow().isoformat(),
        "items": items
    }
    return memory


def build_summary(items):
    by_game = defaultdict(list)

    for it in items:
        game = (it.get("game_name") or "generic").lower()
        by_game[game].append(it)

    summary = {}

    for game, rows in by_game.items():
        captions = [r.get("caption_final", "") for r in rows if r.get("caption_final")]

        hot_words = []
        for c in captions:
            words = c.lower().split()
            hot_words.extend(words)

        # naive frequency
        freq = defaultdict(int)
        for w in hot_words:
            if len(w) < 4:
                continue
            freq[w] += 1

        top_words = sorted(freq.items(), key=lambda x: -x[1])[:10]
        top_words = [w for w, _ in top_words]

        summary[game] = {
            "recent_posts": len(rows),
            "top_words": top_words,
            "patterns_hint": [
                "usar conflicto",
                "usar pregunta final",
                "mencionar juego explícitamente"
            ]
        }

    return {
        "version": "v1",
        "generated_at": datetime.utcnow().isoformat(),
        "games": summary
    }


def run():
    state = load_json(STATE_PATH)

    history = state.get("published_history", [])
    if not history:
        print("No published history found.")
        return

    recents = recent_items(history)
    if not recents:
        print("No recent items.")
        return

    memory = build_editorial_memory(recents)
    summary = build_summary(recents)

    save_json(MEMORY_PATH, memory)
    save_json(SUMMARY_PATH, summary)

    print(f"Editorial memory updated. Items: {len(recents)}")


if __name__ == "__main__":
    run()
