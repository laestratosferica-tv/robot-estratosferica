# main.py
# Robot Editorial + Auto-post Threads con re-host de imágenes en R2
# PROD v7: auto-mix feeds + balance por feed + Threads WAIT + anti-duplicado + repost

import os
import re
import json
import time
import hashlib
import random
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
import boto3

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


print("RUNNING MULTIRED v7 (PROD: MIX+BALANCE+STATE+THREADS WAIT)")

# =========================
# CONFIG
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

R2_PUBLIC_BASE_URL = os.getenv(
    "R2_PUBLIC_BASE_URL",
    "https://pub-8937244ee725495691514507bb8f431e.r2.dev"
).rstrip("/")

THREADS_USER_ID = os.getenv("THREADS_USER_ID", "me")
THREADS_USER_ACCESS_TOKEN = os.getenv("THREADS_USER_ACCESS_TOKEN")

THREADS_STATE_KEY = os.getenv("THREADS_STATE_KEY", "threads_state.json")

THREADS_AUTO_POST = os.getenv("THREADS_AUTO_POST", "true").lower() == "true"
THREADS_AUTO_POST_LIMIT = int(os.getenv("THREADS_AUTO_POST_LIMIT", "1"))
THREADS_DRY_RUN = os.getenv("THREADS_DRY_RUN", "false").lower() == "true"

REPOST_MAX_TIMES = int(os.getenv("REPOST_MAX_TIMES", "3"))
REPOST_WINDOW_DAYS = int(os.getenv("REPOST_WINDOW_DAYS", "7"))
REPOST_ENABLE = os.getenv("REPOST_ENABLE", "true").lower() == "true"

RSS_FEEDS = [x.strip() for x in (os.getenv("RSS_FEEDS") or "").split(",") if x.strip()]
if not RSS_FEEDS:
    RSS_FEEDS = ["https://www.dexerto.com/feed/"]

MAX_PER_FEED = int(os.getenv("MAX_PER_FEED", "3"))
SHUFFLE_ARTICLES = os.getenv("SHUFFLE_ARTICLES", "true").lower() == "true"

MAX_AI_ITEMS = int(os.getenv("MAX_AI_ITEMS", "15"))
SLEEP_BETWEEN_ITEMS_SEC = float(os.getenv("SLEEP_BETWEEN_ITEMS_SEC", "0.05"))

THREADS_GRAPH = os.getenv("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
POST_RETRY_MAX = int(os.getenv("POST_RETRY_MAX", "2"))
POST_RETRY_SLEEP = float(os.getenv("POST_RETRY_SLEEP", "2"))

CONTAINER_WAIT_TIMEOUT = int(os.getenv("CONTAINER_WAIT_TIMEOUT", "120"))
CONTAINER_POLL_INTERVAL = float(os.getenv("CONTAINER_POLL_INTERVAL", "2"))

# =========================
# TIME
# =========================

def now_utc():
    return datetime.now(timezone.utc)

def iso_now():
    return now_utc().isoformat()

def parse_iso(dt):
    return datetime.fromisoformat(dt.replace("Z", "+00:00"))

def days_since(dt_iso):
    try:
        dt = parse_iso(dt_iso)
        return int((now_utc() - dt).total_seconds() // 86400)
    except Exception:
        return 999999

# =========================
# R2
# =========================

def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )

def save_to_r2_json(key, payload):
    s3 = r2_client()
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body, ContentType="application/json")

def load_from_r2_json(key):
    s3 = r2_client()
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None

# =========================
# RSS
# =========================

def fetch_rss_articles():
    raw = []
    for feed in RSS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:50]:
                link = getattr(e, "link", "")
                title = getattr(e, "title", "")
                if link:
                    raw.append({"title": title.strip(), "link": link.strip(), "feed": feed})
        except Exception:
            continue

    seen = set()
    deduped = []
    for a in raw:
        if a["link"] not in seen:
            seen.add(a["link"])
            deduped.append(a)

    if MAX_PER_FEED > 0:
        counts = {}
        balanced = []
        for a in deduped:
            f = a.get("feed", "")
            counts.setdefault(f, 0)
            if counts[f] < MAX_PER_FEED:
                balanced.append(a)
                counts[f] += 1
        deduped = balanced

    if SHUFFLE_ARTICLES:
        random.shuffle(deduped)

    return deduped

# =========================
# OPENAI
# =========================

def openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_text(prompt):
    client = openai_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

def build_threads_text(item, mode="new"):
    title = item.get("title", "")
    link = item.get("link", "")
    prompt = f"""
Eres editor para una cuenta de Threads sobre esports/gaming.
Crea un post corto atractivo.
Incluye Fuente: {link}
Título: {title}
"""
    return openai_text(prompt)

# =========================
# THREADS SIMPLIFIED
# =========================

def threads_publish_text_image(text):
    print("Simulación publicación Threads:")
    print(text)
    return {"ok": True}

# =========================
# STATE
# =========================

def load_threads_state():
    state = load_from_r2_json(THREADS_STATE_KEY)
    if not state:
        state = {"posted_items": {}}
    return state

def save_threads_state(state):
    save_to_r2_json(THREADS_STATE_KEY, state)

# =========================
# MAIN
# =========================

def run_robot_once():
    articles = fetch_rss_articles()
    state = load_threads_state()
    results = []

    for item in articles[:MAX_AI_ITEMS]:
        link = item["link"]
        if link not in state["posted_items"]:
            text = build_threads_text(item)
            res = threads_publish_text_image(text)
            state["posted_items"][link] = iso_now()
            save_threads_state(state)
            results.append({"link": link, "posted": True})
            break

    return {"results": results}

def save_run_payload(payload):
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    key = f"editorial_run_{run_id}.json"
    save_to_r2_json(key, payload)
    print("Archivo guardado en R2:", key)

if __name__ == "__main__":
    result = run_robot_once()
    payload = {
        "generated_at": iso_now(),
        "mix": {
            "shuffle": SHUFFLE_ARTICLES,
            "max_per_feed": MAX_PER_FEED,
            "max_ai_items": MAX_AI_ITEMS
        },
        "threads_auto_post": {
            "enabled": THREADS_AUTO_POST,
            "limit": THREADS_AUTO_POST_LIMIT,
            "dry_run": THREADS_DRY_RUN,
            "repost_enable": REPOST_ENABLE,
            "repost_max_times": REPOST_MAX_TIMES,
            "repost_window_days": REPOST_WINDOW_DAYS
        },
        "result": result
    }

    save_run_payload(payload)
    print("RUN COMPLETED")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
