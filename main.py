import os
import re
import json
import time
import hashlib
import random
from datetime import datetime, timezone
from urllib.parse import urljoin
from typing import Optional, Tuple, Dict, Any, List

import requests
import boto3

# RSS
try:
    import feedparser
except Exception:
    feedparser = None

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Google Trends (pytrends) - opcional pero recomendado
try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None


print("RUNNING MEDIA ENGINE (Threads REAL + IG Queue + Multi-account via accounts.json)")

# =========================
# GLOBAL ENV (infra)
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

THREADS_GRAPH = os.getenv("THREADS_GRAPH", "https://graph.threads.net").rstrip("/")
THREADS_USER_ACCESS_TOKEN = os.getenv("THREADS_USER_ACCESS_TOKEN")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
POST_RETRY_MAX = int(os.getenv("POST_RETRY_MAX", "2"))
POST_RETRY_SLEEP = float(os.getenv("POST_RETRY_SLEEP", "2"))

CONTAINER_WAIT_TIMEOUT = int(os.getenv("CONTAINER_WAIT_TIMEOUT", "120"))
CONTAINER_POLL_INTERVAL = float(os.getenv("CONTAINER_POLL_INTERVAL", "2"))

IG_CAROUSEL_MAX_IMAGES = int(os.getenv("IG_CAROUSEL_MAX_IMAGES", "5"))

# Verificación anti-humo (2 fuentes RSS distintas)
VERIFY_NEWS = os.getenv("VERIFY_NEWS", "true").lower() == "true"
VERIFY_MIN_OVERLAP = float(os.getenv("VERIFY_MIN_OVERLAP", "0.45"))

# Tendencias (LATAM proxy)
ENABLE_TRENDS = os.getenv("ENABLE_TRENDS", "true").lower() == "true"
TRENDS_COUNTRY = os.getenv("TRENDS_COUNTRY", "mexico")  # mexico | argentina | colombia | chile | peru etc.
TRENDS_TOP_N = int(os.getenv("TRENDS_TOP_N", "10"))

# =========================
# TIME
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
# HTTP helpers
# =========================

def _threads_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def _raise_meta_error(r: requests.Response, label: str = "HTTP") -> None:
    if r.status_code < 400:
        return

    print(f"\n====== {label} ERROR ======")
    print("URL:", r.request.url)
    print("METHOD:", r.request.method)
    if r.request.body:
        body = r.request.body
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        print("REQUEST BODY:", body)
    print("STATUS:", r.status_code)
    print("RESPONSE TEXT:", r.text)
    print("========================\n")
    r.raise_for_status()

def _post_with_retries(url: str, *, headers=None, data=None, params=None, label: str = "HTTP POST") -> requests.Response:
    last_err = None
    for attempt in range(POST_RETRY_MAX + 1):
        try:
            r = requests.post(url, headers=headers, data=data, params=params, timeout=HTTP_TIMEOUT)
            _raise_meta_error(r, label)
            return r
        except Exception as e:
            last_err = e
            if attempt < POST_RETRY_MAX:
                time.sleep(POST_RETRY_SLEEP)
            else:
                raise
    raise last_err  # type: ignore[misc]

# =========================
# R2 (S3)
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

def save_to_r2_json(key: str, payload: dict) -> None:
    s3 = r2_client()
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body, ContentType="application/json")

def load_from_r2_json(key: str):
    s3 = r2_client()
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data)
    except Exception:
        return None

def _guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    return ".jpg"

def upload_bytes_to_r2_public(image_bytes: bytes, ext: str, prefix: str) -> str:
    s3 = r2_client()
    h = hashlib.sha1(image_bytes).hexdigest()[:16]
    key = f"{prefix}/{h}{ext}"

    content_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType=content_type,
    )

    url = f"{R2_PUBLIC_BASE_URL}/{key}"

    head = requests.head(url, timeout=10, allow_redirects=True)
    if head.status_code != 200:
        raise RuntimeError(f"R2 public URL no accesible (status {head.status_code}): {url}")

    ct = (head.headers.get("Content-Type") or "").lower()
    if "image" not in ct:
        raise RuntimeError(f"R2 URL no parece imagen (Content-Type={ct}): {url}")

    return url

# =========================
# RSS / Images extraction
# =========================

META_IMAGE_RE = re.compile(
    r'<meta\s+(?:property|name)=["\'](og:image|twitter:image)["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE
)
META_IMAGE_RE2 = re.compile(
    r'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\'](og:image|twitter:image)["\']',
    re.IGNORECASE
)
IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)

def extract_best_images(page_url: str, max_images: int = 5) -> List[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(page_url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        html = r.text

        found: List[str] = []

        metas: List[Tuple[str, str]] = []
        for m in META_IMAGE_RE.finditer(html):
            metas.append((m.group(1).lower(), m.group(2).strip()))
        for m in META_IMAGE_RE2.finditer(html):
            metas.append((m.group(2).lower(), m.group(1).strip()))

        for k in ["og:image", "twitter:image"]:
            for (name, img) in metas:
                if name == k and img:
                    if img.startswith("/"):
                        img = urljoin(page_url, img)
                    if img not in found:
                        found.append(img)

        if len(found) < max_images:
            for m in IMG_SRC_RE.finditer(html):
                img = m.group(1).strip()
                if not img:
                    continue
                if img.startswith("/"):
                    img = urljoin(page_url, img)
                if img.lower().startswith("data:"):
                    continue
                if img not in found:
                    found.append(img)
                if len(found) >= max_images:
                    break

        return found[:max_images]
    except Exception:
        return []

def download_image_bytes(image_url: str) -> Tuple[bytes, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": image_url,
    }
    r = requests.get(image_url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    ext = _guess_ext_from_content_type(r.headers.get("Content-Type", ""))
    return r.content, ext

def fetch_rss_articles(rss_feeds: List[str], max_per_feed: int, shuffle: bool) -> List[Dict[str, Any]]:
    if not feedparser:
        raise RuntimeError("Falta feedparser. Agrégalo a requirements.txt: feedparser")

    raw: List[Dict[str, Any]] = []
    for feed in rss_feeds:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:50]:
                link = getattr(e, "link", None) or ""
                title = getattr(e, "title", "") or ""
                published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
                if link:
                    raw.append({
                        "title": title.strip(),
                        "link": link.strip(),
                        "published": published,
                        "feed": feed,
                    })
        except Exception:
            continue

    # de-dup por link
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for a in raw:
        if a["link"] not in seen:
            seen.add(a["link"])
            deduped.append(a)

    # balance por feed
    if max_per_feed > 0:
        counts: Dict[str, int] = {}
        balanced: List[Dict[str, Any]] = []
        for a in deduped:
            f = a.get("feed") or "unknown"
            counts.setdefault(f, 0)
            if counts[f] < max_per_feed:
                balanced.append(a)
                counts[f] += 1
        deduped = balanced

    if shuffle:
        random.shuffle(deduped)

    return deduped

# =========================
# Trending (Google Trends)
# =========================

def get_trending_keywords_latam(top_n: int = 10, country: str = "mexico") -> List[str]:
    """
    Retorna keywords trending del país (proxy LATAM).
    Si falla o no está pytrends instalado, retorna [].
    """
    if not TrendReq:
        return []
    try:
        pytrend = TrendReq(hl='es-419', tz=0)
        trending = pytrend.trending_searches(pn=country)
        keywords = trending[0].tolist()
        return [k for k in keywords if isinstance(k, str)][:top_n]
    except Exception as e:
        print("Google Trends error:", str(e))
        return []

# =========================
# Verification (anti-humo)
# =========================

def normalize_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9áéíóúñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def title_tokens(t: str) -> set:
    stop = {
        "the","a","an","and","or","to","of","in","on","for","with","vs",
        "de","la","el","y","en","para","con","un","una","del","al"
    }
    toks = set([x for x in normalize_title(t).split() if len(x) > 2 and x not in stop])
    return toks

def is_verified_by_rss(item: Dict[str, Any], all_items: List[Dict[str, Any]], min_overlap: float = 0.45) -> bool:
    """
    Verifica si el título aparece "similar" en otro feed distinto.
    """
    t1 = title_tokens(item.get("title",""))
    if not t1:
        return False
    feed1 = item.get("feed")
    for other in all_items:
        if other is item:
            continue
        if other.get("feed") == feed1:
            continue
        t2 = title_tokens(other.get("title",""))
        if not t2:
            continue
        overlap = len(t1 & t2) / max(1, len(t1))
        if overlap >= min_overlap:
            return True
    return False

# =========================
# OpenAI (text)
# =========================

def openai_client():
    if not OpenAI:
        raise RuntimeError("No se pudo importar OpenAI. Revisa requirements.txt (openai).")
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_text(prompt: str) -> str:
    client = openai_client()
    try:
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

def build_threads_text(item: Dict[str, Any], mode: str = "new") -> str:
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
    if "Fuente:" not in text:
        text = f"{text}\n\nFuente: {link}"
    return text.strip()

def build_instagram_caption(item: Dict[str, Any], mode: str, link: str) -> str:
    title = item.get("title", "")
    prompt = f"""
Eres editor de Instagram (esports/gaming) para público en ESPAÑOL LATAM.
Escribe un caption natural y humano:
- 1-2 párrafos cortos
- 5-10 hashtags relevantes al final
- Cierra con una pregunta
- Incluye "Fuente:" + link al final
Título: {title}
Link: {link}
"""
    text = openai_text(prompt).strip()
    if "Fuente:" not in text:
        text = f"{text}\n\nFuente: {link}"
    return text

# =========================
# Threads API (real)
# =========================

def threads_create_container_image(user_id: str, access_token: str, text: str, image_url: str) -> str:
    url = f"{THREADS_GRAPH}/{user_id}/threads"
    data = {"media_type": "IMAGE", "image_url": image_url, "text": text}
    r = _post_with_retries(url, headers=_threads_headers(access_token), data=data, label="THREADS CREATE_CONTAINER")
    return r.json()["id"]

def threads_wait_container(container_id: str, access_token: str, timeout_sec: Optional[int] = None) -> Dict[str, Any]:
    if timeout_sec is None:
        timeout_sec = CONTAINER_WAIT_TIMEOUT

    url = f"{THREADS_GRAPH}/{container_id}"
    start = time.time()
    last = None

    while time.time() - start < timeout_sec:
        r = requests.get(
            url,
            headers=_threads_headers(access_token),
            params={"fields": "status,error_message"},
            timeout=HTTP_TIMEOUT
        )
        _raise_meta_error(r, "THREADS CONTAINER STATUS")

        j = r.json()
        last = j
        status = j.get("status")

        if status == "FINISHED":
            return j
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"Container failed: {j}")

        time.sleep(CONTAINER_POLL_INTERVAL)

    raise TimeoutError(f"Container not ready after {timeout_sec}s: {last}")

def threads_publish(user_id: str, access_token: str, container_id: str) -> Dict[str, Any]:
    url = f"{THREADS_GRAPH}/{user_id}/threads_publish"
    r = _post_with_retries(
        url,
        headers=_threads_headers(access_token),
        data={"creation_id": container_id},
        label="THREADS PUBLISH"
    )
    return r.json()

def threads_publish_text_image(user_id: str, access_token: str, dry_run: bool, text: str, image_url_from_news: str, threads_media_prefix: str) -> Dict[str, Any]:
    if dry_run:
        print("[DRY_RUN] Threads post:", text)
        print("[DRY_RUN] Image source:", image_url_from_news)
        return {"ok": True, "dry_run": True}

    if not access_token:
        raise RuntimeError("Falta THREADS_USER_ACCESS_TOKEN")

    img_bytes, ext = download_image_bytes(image_url_from_news)
    r2_url = upload_bytes_to_r2_public(img_bytes, ext, prefix=threads_media_prefix)
    print("IMAGE re-hosted on R2:", r2_url)

    container_id = threads_create_container_image(user_id, access_token, text, r2_url)
    print("Container created:", container_id)

    threads_wait_container(container_id, access_token)

    res = threads_publish(user_id, access_token, container_id)
    print("Threads publish response:", res)

    return {"ok": True, "container": {"id": container_id}, "publish": res, "image_url": r2_url}

# =========================
# State
# =========================

def load_threads_state(state_key: str) -> Dict[str, Any]:
    state = load_from_r2_json(state_key)
    if not state:
        state = {"posted_items": {}, "posted_links": [], "last_posted_at": None}
    state.setdefault("posted_items", {})
    state.setdefault("posted_links", [])
    state.setdefault("last_posted_at", None)
    return state

def save_threads_state(state_key: str, state: Dict[str, Any]) -> None:
    save_to_r2_json(state_key, state)

def mark_posted(state: Dict[str, Any], link: str) -> None:
    pi = state["posted_items"].get(link, {"times": 0, "last_posted_at": None})
    pi["times"] = int(pi.get("times", 0)) + 1
    pi["last_posted_at"] = iso_now()
    state["posted_items"][link] = pi
    if link not in state["posted_links"]:
        state["posted_links"].append(link)
    state["posted_links"] = state["posted_links"][-400:]
    state["last_posted_at"] = iso_now()

def is_new_allowed(state: Dict[str, Any], link: str) -> bool:
    return link not in state["posted_items"]

def repost_eligible(state: Dict[str, Any], link: str, repost_max_times: int, repost_window_days: int) -> bool:
    pi = state["posted_items"].get(link)
    if not pi:
        return False
    times = int(pi.get("times", 0))
    if times >= repost_max_times:
        return False
    last = pi.get("last_posted_at")
    if not last:
        return True
    return days_since(last) >= repost_window_days

def pick_item(articles: List[Dict[str, Any]], state: Dict[str, Any], repost_enable: bool, repost_max_times: int, repost_window_days: int) -> Tuple[Optional[Dict[str, Any]], str]:
    for a in articles:
        if a.get("link") and is_new_allowed(state, a["link"]):
            return a, "new"
    if repost_enable:
        for a in articles:
            if a.get("link") and a["link"] in state["posted_items"] and repost_eligible(state, a["link"], repost_max_times, repost_window_days):
                return a, "repost"
    return None, "none"

# =========================
# IG queue helpers
# =========================

def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def choose_ig_format(has_video: bool, image_count: int) -> str:
    if has_video:
        return "reel"
    if image_count >= 2:
        return "carousel"
    return "image"

def save_ig_queue_item(prefix: str, payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    h = _short_hash(s + iso_now())
    key = f"{prefix}/{now_utc().strftime('%Y%m%d_%H%M%S')}_{h}.json"
    save_to_r2_json(key, payload)
    return key

# =========================
# Accounts loading
# =========================

def load_accounts() -> List[Dict[str, Any]]:
    """
    Si existe 'accounts.json' en el repo, lo usa.
    Si no existe, corre 1 cuenta default (estratosferica).
    """
    try:
        with open("accounts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "accounts" in data and isinstance(data["accounts"], list):
            return data["accounts"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Default 1 cuenta (fuentes mejoradas)
    return [{
        "account_id": "estratosferica",
        "rss_feeds": [
            # Gaming / esports
            "https://www.dexerto.com/feed/",
            "https://dotesports.com/feed",
            "https://www.videogameschronicle.com/feed/",
            "https://www.eurogamer.net/feed",
            "https://www.pcgamer.com/rss/",
            "https://www.polygon.com/rss/index.xml",
            # Tech / Hardware (gamer)
            "https://www.theverge.com/rss/index.xml",
            "https://www.tomshardware.com/feeds/all",
        ],
        "max_per_feed": int(os.getenv("MAX_PER_FEED", "3")),
        "shuffle_articles": os.getenv("SHUFFLE_ARTICLES", "true").lower() == "true",
        "max_ai_items": int(os.getenv("MAX_AI_ITEMS", "20")),
        "threads": {
            "user_id": os.getenv("THREADS_USER_ID", "me"),
            "state_key": "accounts/estratosferica/threads_state.json",
            "auto_post": os.getenv("THREADS_AUTO_POST", "true").lower() == "true",
            "auto_post_limit": int(os.getenv("THREADS_AUTO_POST_LIMIT", "1")),
            "dry_run": os.getenv("THREADS_DRY_RUN", "false").lower() == "true",
            "repost_enable": os.getenv("REPOST_ENABLE", "true").lower() == "true",
            "repost_max_times": int(os.getenv("REPOST_MAX_TIMES", "3")),
            "repost_window_days": int(os.getenv("REPOST_WINDOW_DAYS", "7"))
        },
        "r2": {
            "threads_media_prefix": "threads_media/estratosferica",
            "ig_queue_prefix": "ugc/ig_queue/estratosferica"
        }
    }]

# =========================
# Main per account run
# =========================

def run_account(cfg: Dict[str, Any]) -> Dict[str, Any]:
    account_id = cfg.get("account_id", "unknown")
    print(f"\n===== RUN ACCOUNT: {account_id} =====")

    rss_feeds = cfg.get("rss_feeds") or []
    max_per_feed = int(cfg.get("max_per_feed", 3))
    shuffle = bool(cfg.get("shuffle_articles", True))
    max_ai_items = int(cfg.get("max_ai_items", 15))

    threads_cfg = cfg.get("threads", {})
    threads_user_id = threads_cfg.get("user_id", "me")
    state_key = threads_cfg.get("state_key", f"accounts/{account_id}/threads_state.json")
    auto_post = bool(threads_cfg.get("auto_post", True))
    auto_post_limit = int(threads_cfg.get("auto_post_limit", 1))
    dry_run = bool(threads_cfg.get("dry_run", False))
    repost_enable = bool(threads_cfg.get("repost_enable", True))
    repost_max_times = int(threads_cfg.get("repost_max_times", 3))
    repost_window_days = int(threads_cfg.get("repost_window_days", 7))

    r2_cfg = cfg.get("r2", {})
    threads_media_prefix = r2_cfg.get("threads_media_prefix", f"threads_media/{account_id}").strip().strip("/")
    ig_queue_prefix = r2_cfg.get("ig_queue_prefix", f"ugc/ig_queue/{account_id}").strip().strip("/")

    # RSS
    print("Obteniendo artículos (RSS)...")
    articles = fetch_rss_articles(rss_feeds, max_per_feed=max_per_feed, shuffle=shuffle)
    print(f"{len(articles)} artículos candidatos tras mix/balance (MAX_PER_FEED={max_per_feed}, SHUFFLE={shuffle})")
    feeds_in_run = sorted(set([a.get("feed", "") for a in articles]))
    print("FEEDS EN ESTA CORRIDA:", feeds_in_run)
    print("TOTAL FEEDS:", len(feeds_in_run))

    # Tendencias: empuja arriba los artículos que contengan keywords trending
    trending_keywords: List[str] = []
    if ENABLE_TRENDS:
        trending_keywords = get_trending_keywords_latam(top_n=TRENDS_TOP_N, country=TRENDS_COUNTRY)
        if trending_keywords:
            print("Trending LATAM:", trending_keywords)
            articles = sorted(
                articles,
                key=lambda a: any(kw.lower() in (a.get("title", "").lower()) for kw in trending_keywords),
                reverse=True
            )
        else:
            print("Trending LATAM: (sin datos / no disponible)")

    processed = []
    for a in articles[:max_ai_items]:
        processed.append(a)
        time.sleep(0.05)

    state = load_threads_state(state_key)
    print("STATE posted_items:", len(state.get("posted_items", {})))

    posted_count = 0
    results = []

    while posted_count < auto_post_limit:
        item, mode = pick_item(processed, state, repost_enable, repost_max_times, repost_window_days)
        if not item:
            print("No hay item nuevo ni repost elegible.")
            break

        link = item["link"]
        label = "NUEVO" if mode == "new" else "REPOST"
        print(f"Seleccionado ({label}): {link}")

        # Anti-humo: exige confirmación por otra fuente RSS
        if VERIFY_NEWS:
            if not is_verified_by_rss(item, processed, min_overlap=VERIFY_MIN_OVERLAP):
                print("No verificado por segunda fuente RSS. Saltando para evitar humo:", link)
                processed = [x for x in processed if x.get("link") != link]
                continue

        text = build_threads_text(item, mode=mode)

        # Imagen para Threads (al menos 1)
        imgs_1 = extract_best_images(link, max_images=1)
        if not imgs_1:
            print("No se encontró imagen (og/twitter/img). Se omite para evitar post sin imagen.")
            processed = [x for x in processed if x.get("link") != link]
            continue

        if not auto_post:
            print("THREADS_AUTO_POST desactivado. (No se publica)")
            results.append({"link": link, "mode": mode, "posted": False, "reason": "auto_post_off"})
            break

        print(f"Publicando en Threads ({label})...")
        try:
            threads_res = threads_publish_text_image(
                user_id=threads_user_id,
                access_token=THREADS_USER_ACCESS_TOKEN,
                dry_run=dry_run,
                text=text,
                image_url_from_news=imgs_1[0],
                threads_media_prefix=threads_media_prefix,
            )

            mark_posted(state, link)
            save_threads_state(state_key, state)

            posted_count += 1
            results.append({"link": link, "mode": mode, "posted": True, "threads": threads_res})
            print("Auto-post Threads: OK ✅")

            # IG queue (reels>carousel>image) - reels solo si hay video (por ahora no inventamos)
            try:
                candidates = extract_best_images(link, max_images=IG_CAROUSEL_MAX_IMAGES)

                r2_images: List[str] = []
                for img_url in candidates:
                    try:
                        b, ext = download_image_bytes(img_url)
                        r2_img = upload_bytes_to_r2_public(b, ext, prefix=threads_media_prefix)
                        if r2_img not in r2_images:
                            r2_images.append(r2_img)
                    except Exception:
                        continue
                    if len(r2_images) >= IG_CAROUSEL_MAX_IMAGES:
                        break

                threads_img = threads_res.get("image_url") if isinstance(threads_res, dict) else None
                if not r2_images and threads_img:
                    r2_images = [threads_img]

                has_video = False
                reel_video_url = None

                ig_format = choose_ig_format(has_video=has_video, image_count=len(r2_images))
                ig_caption = build_instagram_caption(item, mode=mode, link=link)

                ig_payload = {
                    "created_at": iso_now(),
                    "account_id": account_id,
                    "source": {
                        "link": link,
                        "feed": item.get("feed"),
                        "mode": mode,
                        "title": item.get("title"),
                    },
                    "format": ig_format,  # reel | carousel | image
                    "caption": ig_caption,
                    "assets": {
                        "images": r2_images[:IG_CAROUSEL_MAX_IMAGES],
                        "reel_video_url": reel_video_url,
                    },
                    "threads": {
                        "publish_id": (threads_res.get("publish") or {}).get("id") if isinstance(threads_res, dict) else None,
                        "image_url": threads_img,
                    },
                }

                ig_key = save_ig_queue_item(ig_queue_prefix, ig_payload)
                print("IG queue guardado en R2:", ig_key)
            except Exception as e:
                print("IG queue: falló (no rompe el run):", str(e))

        except Exception as e:
            print("Auto-post Threads: FALLÓ ❌")
            print("ERROR:", str(e))
            results.append({"link": link, "mode": mode, "posted": False, "error": str(e)})
            break

        processed = [x for x in processed if x.get("link") != link]

    run_payload = {
        "generated_at": iso_now(),
        "account_id": account_id,
        "mix": {
            "shuffle": shuffle,
            "max_per_feed": max_per_feed,
            "max_ai_items": max_ai_items,
        },
        "trends": {
            "enabled": ENABLE_TRENDS,
            "country": TRENDS_COUNTRY,
            "top_n": TRENDS_TOP_N,
            "keywords": trending_keywords,
        },
        "verification": {
            "enabled": VERIFY_NEWS,
            "min_overlap": VERIFY_MIN_OVERLAP,
        },
        "result": {
            "posted_count": posted_count,
            "results": results,
        }
    }
    return run_payload

def save_run_payload(account_id: str, payload: Dict[str, Any]) -> str:
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    key = f"accounts/{account_id}/runs/editorial_run_{run_id}.json"
    save_to_r2_json(key, payload)
    print("Archivo guardado en R2:", key)
    return key

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    accounts = load_accounts()

    all_results = []
    for cfg in accounts:
        account_id = cfg.get("account_id", "unknown")
        payload = run_account(cfg)
        run_key = save_run_payload(account_id, payload)
        all_results.append({"account_id": account_id, "run_key": run_key, "payload": payload})
        print("RUN COMPLETED:", account_id)

    print("\n===== SUMMARY =====")
    print(json.dumps({"runs": all_results}, ensure_ascii=False, indent=2))
