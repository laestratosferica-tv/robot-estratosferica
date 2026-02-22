# main.py
# Robot Editorial + Auto-post Threads con re-host de imágenes en R2
# Ajustes: n=3 reposts, ventana=7 días

import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin

import requests
import boto3

# Opcional pero recomendado para RSS
try:
    import feedparser
except Exception:
    feedparser = None

# OpenAI (según tu repo ya lo estás usando)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

print("RUNNING MULTIRED v1 (is_gaming + category)")

# =========================
# CONFIG
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")  # ej: https://xxxxx.r2.cloudflarestorage.com
BUCKET_NAME = os.getenv("BUCKET_NAME")

THREADS_USER_ID = os.getenv("THREADS_USER_ID", "25864040949913281")
THREADS_USER_ACCESS_TOKEN = os.getenv("THREADS_USER_ACCESS_TOKEN")

THREADS_STATE_KEY = os.getenv("THREADS_STATE_KEY", "threads_state.json")

THREADS_AUTO_POST = os.getenv("THREADS_AUTO_POST", "true").lower() == "true"
THREADS_AUTO_POST_LIMIT = int(os.getenv("THREADS_AUTO_POST_LIMIT", "1"))
THREADS_DRY_RUN = os.getenv("THREADS_DRY_RUN", "false").lower() == "true"

# Repost logic
REPOST_MAX_TIMES = int(os.getenv("REPOST_MAX_TIMES", "3"))  # n=3
REPOST_WINDOW_DAYS = int(os.getenv("REPOST_WINDOW_DAYS", "7"))  # ventana=7
REPOST_ENABLE = os.getenv("REPOST_ENABLE", "true").lower() == "true"

# RSS feeds (separados por coma). Si no pones nada, usa un ejemplo.
RSS_FEEDS = [x.strip() for x in (os.getenv("RSS_FEEDS") or "").split(",") if x.strip()]
if not RSS_FEEDS:
    RSS_FEEDS = [
        # Pon aquí tus feeds reales (o en variable RSS_FEEDS)
        "https://www.dexerto.com/feed/",
    ]

# Para no reventar rate limits
MAX_AI_ITEMS = int(os.getenv("MAX_AI_ITEMS", "5"))
SLEEP_BETWEEN_ITEMS_SEC = float(os.getenv("SLEEP_BETWEEN_ITEMS_SEC", "0.25"))

THREADS_GRAPH = os.getenv("THREADS_GRAPH", "https://graph.facebook.com/v20.0")


# =========================
# HELPERS: TIME
# =========================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_now() -> str:
    return now_utc().isoformat()

def parse_iso(dt: str) -> datetime:
    return datetime.fromisoformat(dt.replace("Z", "+00:00"))

def days_since(dt_iso: str) -> int:
    try:
        dt = parse_iso(dt_iso)
        return int((now_utc() - dt).total_seconds() // 86400)
    except Exception:
        return 999999


# =========================
# HELPERS: R2 (S3)
# =========================

def r2_client():
    if not (R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise RuntimeError("Faltan credenciales R2/S3 (R2_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )

def save_to_r2(key: str, payload: dict):
    s3 = r2_client()
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body, ContentType="application/json")

def load_from_r2(key: str):
    s3 = r2_client()
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data)
    except s3.exceptions.NoSuchKey:
        return None
    except Exception:
        return None

def r2_public_base_url_from_endpoint(endpoint_url: str) -> str:
    # Convierte: https://xxxxx.r2.cloudflarestorage.com -> https://xxxxx.r2.dev
    # Si usas dominio propio público, reemplaza esta lógica por tu dominio.
    return endpoint_url.replace(".r2.cloudflarestorage.com", ".r2.dev").rstrip("/")

def upload_bytes_to_r2_public(image_bytes: bytes, ext: str, prefix="threads_media") -> str:
    s3 = r2_client()
    h = hashlib.sha1(image_bytes).hexdigest()[:16]
    key = f"{prefix}/{h}{ext}"

    content_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext.lower(), "image/jpeg")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType=content_type,
    )

    base = r2_public_base_url_from_endpoint(R2_ENDPOINT_URL)
    # forma común de r2.dev: /<bucket>/<key>
    return f"{base}/{BUCKET_NAME}/{key}"


# =========================
# RSS / ARTÍCULOS
# =========================

def fetch_rss_articles():
    if not feedparser:
        raise RuntimeError("Falta feedparser. Agrégalo a requirements.txt: feedparser")

    articles = []
    for feed in RSS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:50]:
                link = getattr(e, "link", None) or ""
                title = getattr(e, "title", "") or ""
                published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
                articles.append({
                    "title": title.strip(),
                    "link": link.strip(),
                    "published": published,
                    "feed": feed,
                })
        except Exception:
            continue

    # dedupe por link
    seen = set()
    out = []
    for a in articles:
        if a["link"] and a["link"] not in seen:
            seen.add(a["link"])
            out.append(a)
    return out


# =========================
# EXTRAER IMAGEN OG
# =========================

OG_IMAGE_RE = re.compile(r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE)
OG_IMAGE_RE2 = re.compile(r'content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', re.IGNORECASE)

def extract_og_image(url: str) -> str | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        html = r.text
        m = OG_IMAGE_RE.search(html) or OG_IMAGE_RE2.search(html)
        if not m:
            return None
        img = m.group(1).strip()
        # normaliza relativas
        if img.startswith("/"):
            img = urljoin(url, img)
        return img
    except Exception:
        return None

def guess_ext_from_content_type(ct: str):
    ct = (ct or "").lower()
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    return ".jpg"

def download_image_bytes(image_url: str) -> tuple[bytes, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": image_url,
    }
    r = requests.get(image_url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    ext = guess_ext_from_content_type(r.headers.get("Content-Type", ""))
    return r.content, ext


# =========================
# OPENAI: GENERAR TEXTO
# =========================

def openai_client():
    if not OpenAI:
        raise RuntimeError("No se pudo importar OpenAI. Revisa requirements.txt (openai).")
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_text(prompt: str) -> str:
    client = openai_client()

    # Intento 1: responses API (más nueva)
    try:
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
        )
        # resp.output_text existe en SDKs nuevos
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass

    # Intento 2: chat.completions (fallback)
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")

def build_threads_text(item: dict, mode: str = "new") -> str:
    # mode: "new" o "repost"
    title = item.get("title", "")
    link = item.get("link", "")

    if mode == "repost":
        prompt = f"""
Eres editor para una cuenta de Threads sobre esports/gaming.
Reescribe este post como un REPOST con otra mirada/opinión, sin sonar repetido.
- 1 párrafo corto, máximo 260 caracteres si es posible.
- Termina con una pregunta para la comunidad.
- Incluye "Fuente:" + link.
Datos:
Título: {title}
Link: {link}
"""
    else:
        prompt = f"""
Eres editor para una cuenta de Threads sobre esports/gaming.
Crea un post:
- 1 párrafo corto (máximo 260-320 caracteres).
- Termina con una pregunta a la comunidad.
- Incluye "Fuente:" + link.
Datos:
Título: {title}
Link: {link}
"""
    text = openai_text(prompt)

    # Seguridad: asegura que el link aparezca
    if "Fuente:" not in text:
        text = f"{text}\n\nFuente: {link}"
    return text.strip()


# =========================
# THREADS API
# =========================

def threads_create_container_image(user_id, access_token, text, image_url):
    url = f"{THREADS_GRAPH}/{user_id}/threads"
    data = {
        "media_type": "IMAGE",
        "image_url": image_url,
        "text": text,
        "access_token": access_token,
    }
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def threads_wait_container(container_id, access_token, timeout_sec=90):
    url = f"{THREADS_GRAPH}/{container_id}"
    start = time.time()
    last = None
    while time.time() - start < timeout_sec:
        r = requests.get(
            url,
            params={"fields": "status,error_message", "access_token": access_token},
            timeout=30
        )
        r.raise_for_status()
        j = r.json()
        last = j
        status = j.get("status")
        if status == "FINISHED":
            return j
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"Container failed: {j}")
        time.sleep(2)
    raise TimeoutError(f"Container not ready after {timeout_sec}s: {last}")

def threads_publish(user_id, access_token, container_id):
    url = f"{THREADS_GRAPH}/{user_id}/threads_publish"
    r = requests.post(url, data={"creation_id": container_id, "access_token": access_token}, timeout=30)
    r.raise_for_status()
    return r.json()


def threads_publish_text_image(text: str, image_url_from_news: str):
    if THREADS_DRY_RUN:
        print("[DRY_RUN] Threads post:", text)
        print("[DRY_RUN] Image:", image_url_from_news)
        return {"ok": True, "dry_run": True}

    # 1) descargar imagen
    img_bytes, ext = download_image_bytes(image_url_from_news)

    # 2) subir a R2 público
    r2_url = upload_bytes_to_r2_public(img_bytes, ext)
    print("IMAGE re-hosted on R2:", r2_url)

    # 3) container
    container_id = threads_create_container_image(THREADS_USER_ID, THREADS_USER_ACCESS_TOKEN, text, r2_url)
    threads_wait_container(container_id, THREADS_USER_ACCESS_TOKEN, timeout_sec=90)

    # 4) publish
    res = threads_publish(THREADS_USER_ID, THREADS_USER_ACCESS_TOKEN, container_id)
    return {"ok": True, "container": {"id": container_id}, "publish": res, "image_url": r2_url}


# =========================
# STATE
# =========================

def load_threads_state() -> dict:
    state = load_from_r2(THREADS_STATE_KEY)
    if not state:
        state = {
            "posted_items": {},  # link -> {last_posted_at, times}
            "posted_links": [],  # lista (solo para lectura humana / legacy)
            "last_posted_at": None,
        }
    # normaliza
    state.setdefault("posted_items", {})
    state.setdefault("posted_links", [])
    state.setdefault("last_posted_at", None)
    return state

def save_threads_state(state: dict):
    save_to_r2(THREADS_STATE_KEY, state)

def mark_posted(state: dict, link: str):
    pi = state["posted_items"].get(link, {"times": 0, "last_posted_at": None})
    pi["times"] = int(pi.get("times", 0)) + 1
    pi["last_posted_at"] = iso_now()
    state["posted_items"][link] = pi

    # legacy list
    if link not in state["posted_links"]:
        state["posted_links"].append(link)
    state["posted_links"] = state["posted_links"][-200:]
    state["last_posted_at"] = iso_now()


# =========================
# SELECCIÓN: NUEVO vs REPOST
# =========================

def is_new_allowed(state: dict, link: str) -> bool:
    # Si nunca se ha posteado, permitido
    if link not in state["posted_items"]:
        return True
    # si ya se posteó, NO es nuevo
    return False

def repost_eligible(state: dict, link: str) -> bool:
    pi = state["posted_items"].get(link)
    if not pi:
        return False
    times = int(pi.get("times", 0))
    if times >= REPOST_MAX_TIMES:
        return False
    last = pi.get("last_posted_at")
    if not last:
        return True
    # Solo repost si pasaron >= ventana
    return days_since(last) >= REPOST_WINDOW_DAYS

def pick_item(articles: list[dict], state: dict) -> tuple[dict | None, str]:
    # 1) busca un item NUEVO
    for a in articles:
        if a.get("link") and is_new_allowed(state, a["link"]):
            return a, "new"

    # 2) si no hay nuevos, intenta REPOST
    if REPOST_ENABLE:
        for a in articles:
            if a.get("link") and a["link"] in state["posted_items"] and repost_eligible(state, a["link"]):
                return a, "repost"

    return None, "none"


# =========================
# MAIN RUN
# =========================

def run_robot_once():
    # 1) RSS
    print(f"Obteniendo artículos (RSS + filtros)...")
    articles = fetch_rss_articles()
    print(f"{len(articles)} artículos encontrados (filtrados)")

    # 2) IA (limit)
    print(f"Procesando con IA (máximo {MAX_AI_ITEMS})...")
    processed = []
    for a in articles[:MAX_AI_ITEMS]:
        processed.append(a)
        time.sleep(SLEEP_BETWEEN_ITEMS_SEC)

    # 3) Selección final simple
    state = load_threads_state()

    item, mode = pick_item(processed, state)
    if not item:
        print("FINAL ITEMS: 0")
        print("Auto-post Threads: no hay item nuevo (evitando repetidos).")
        # guarda corrida
        return {"final_items": 0, "mode": "none", "posted": False}

    print("FINAL ITEMS: 1")
    link = item["link"]

    # 4) construir texto
    text = build_threads_text(item, mode=mode)

    # 5) conseguir imagen
    og_image = extract_og_image(link)
    if not og_image:
        # si no hay imagen, fallback: post sin imagen NO (para tu caso pediste texto+imagen)
        print("No se encontró og:image. Se omite publicación para evitar post sin imagen.")
        return {"final_items": 1, "mode": mode, "posted": False, "reason": "no_og_image"}

    # 6) autopost
    if THREADS_AUTO_POST and THREADS_AUTO_POST_LIMIT > 0:
        label = "NUEVO" if mode == "new" else "REPOST"
        print(f"Auto-post Threads: publicando 1 post ({label})...")
        try:
            res = threads_publish_text_image(text, og_image)
            print("Threads publish response:", res)
            mark_posted(state, link)
            save_threads_state(state)
            print("Auto-post Threads: OK ✅")
            return {"final_items": 1, "mode": mode, "posted": True, "link": link, "threads": res}
        except Exception as e:
            print("Auto-post Threads: FALLÓ ❌")
            print("ERROR:", str(e))
            return {"final_items": 1, "mode": mode, "posted": False, "error": str(e)}
    else:
        print("Auto-post Threads desactivado.")
        return {"final_items": 1, "mode": mode, "posted": False, "reason": "auto_post_off"}


def save_run_payload(payload: dict):
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    key = f"editorial_run_{run_id}.json"
    save_to_r2(key, payload)
    print("Archivo guardado en R2:", key)


if __name__ == "__main__":
    result = run_robot_once()
    # guarda reporte siempre
    payload = {
        "generated_at": iso_now(),
        "threads_auto_post": {
            "enabled": THREADS_AUTO_POST,
            "limit": THREADS_AUTO_POST_LIMIT,
            "dry_run": THREADS_DRY_RUN,
            "repost_enable": REPOST_ENABLE,
            "repost_max_times": REPOST_MAX_TIMES,
            "repost_window_days": REPOST_WINDOW_DAYS,
            "result": result,
        },
    }
    save_run_payload(payload)
