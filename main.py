# main.py
# Robot Editorial + Auto-post Threads con re-host de imágenes en R2
# FIX DEFINITIVO: Threads API correcta + Debug 400 completo

import os
import re
import json
import time
import hashlib
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


print("RUNNING MULTIRED v2 (THREADS FIXED)")

# =========================
# CONFIG
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# ⚠️ RECOMENDADO: usar "me"
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

MAX_AI_ITEMS = int(os.getenv("MAX_AI_ITEMS", "5"))
SLEEP_BETWEEN_ITEMS_SEC = float(os.getenv("SLEEP_BETWEEN_ITEMS_SEC", "0.25"))

# 🔥 HOST CORRECTO
THREADS_GRAPH = os.getenv("THREADS_GRAPH", "https://graph.threads.net/v20.0")

# =========================
# DEBUG META
# =========================

def _threads_headers(token: str):
    return {"Authorization": f"Bearer {token}"}

def _raise_meta_error(r: requests.Response, label="META"):
    if r.status_code < 400:
        return

    print(f"\n====== {label} ERROR DEBUG ======")
    print("URL:", r.request.url)
    print("STATUS:", r.status_code)
    if r.request.body:
        body = r.request.body
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        print("REQUEST BODY:", body)
    print("RESPONSE TEXT:", r.text)  # 👈 AQUÍ ESTÁ EL JSON COMPLETO
    print("================================\n")

    r.raise_for_status()

# =========================
# TIME
# =========================

def now_utc():
    return datetime.now(timezone.utc)

def iso_now():
    return now_utc().isoformat()

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

R2_PUBLIC_BASE_URL = "https://pub-8937244ee725495691514507bb8f431e.r2.dev"

def upload_bytes_to_r2_public(image_bytes: bytes, ext: str):
    s3 = r2_client()
    h = hashlib.sha1(image_bytes).hexdigest()[:16]
    key = f"threads_media/{h}{ext}"

    content_type = {
        ".png":"image/png",
        ".webp":"image/webp",
        ".jpg":"image/jpeg",
        ".jpeg":"image/jpeg",
    }.get(ext, "image/jpeg")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType=content_type,
    )

    url = f"{R2_PUBLIC_BASE_URL}/{key}"

    # 🔎 Validar que sea accesible públicamente
    head = requests.head(url, timeout=10)
    if head.status_code != 200:
        raise RuntimeError(f"R2 public URL no accesible: {url}")

    return url

# =========================
# RSS
# =========================

def fetch_rss_articles():
    articles = []
    for feed in RSS_FEEDS:
        d = feedparser.parse(feed)
        for e in d.entries[:20]:
            articles.append({
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
            })
    return articles

# =========================
# IMAGEN OG
# =========================

OG_IMAGE_RE = re.compile(r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', re.I)

def extract_og_image(url):
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
    m = OG_IMAGE_RE.search(r.text)
    return m.group(1) if m else None

def download_image_bytes(image_url):
    r = requests.get(image_url, timeout=30)
    r.raise_for_status()
    ct = r.headers.get("Content-Type","").lower()
    if "png" in ct: ext = ".png"
    elif "webp" in ct: ext = ".webp"
    else: ext = ".jpg"
    return r.content, ext

# =========================
# OPENAI
# =========================

def openai_text(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return resp.output_text.strip()

def build_threads_text(item):
    prompt = f"""
Crea un post para Threads sobre gaming.
Máx 280 caracteres.
Termina con pregunta.
Incluye Fuente: {item['link']}
Título: {item['title']}
"""
    return openai_text(prompt)

# =========================
# THREADS API
# =========================

def threads_create_container(text, image_url):
    url = f"{THREADS_GRAPH}/{THREADS_USER_ID}/threads"
    data = {
        "media_type": "IMAGE",
        "image_url": image_url,
        "text": text,
    }

    r = requests.post(url, headers=_threads_headers(THREADS_USER_ACCESS_TOKEN), data=data, timeout=30)
    _raise_meta_error(r, "THREADS CREATE_CONTAINER")
    return r.json()["id"]

def threads_publish(container_id):
    url = f"{THREADS_GRAPH}/{THREADS_USER_ID}/threads_publish"
    r = requests.post(
        url,
        headers=_threads_headers(THREADS_USER_ACCESS_TOKEN),
        data={"creation_id": container_id},
        timeout=30
    )
    _raise_meta_error(r, "THREADS PUBLISH")
    return r.json()

# =========================
# MAIN
# =========================

def run():
    articles = fetch_rss_articles()
    if not articles:
        print("No articles found.")
        return

    item = articles[0]

    text = build_threads_text(item)

    og = extract_og_image(item["link"])
    if not og:
        print("No OG image found.")
        return

    img_bytes, ext = download_image_bytes(og)
    r2_url = upload_bytes_to_r2_public(img_bytes, ext)
    print("IMAGE re-hosted:", r2_url)

    container_id = threads_create_container(text, r2_url)
    print("Container created:", container_id)

    time.sleep(5)

    res = threads_publish(container_id)
    print("THREADS SUCCESS:", res)


if __name__ == "__main__":
    run()
