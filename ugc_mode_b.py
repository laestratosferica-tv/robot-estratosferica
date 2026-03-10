import os
import re
import json
import time
import requests
import boto3


# =========================
# ENV HELPERS
# =========================

def env_nonempty(name, default=None):
    v = os.getenv(name)
    if not v:
        return default
    v = v.strip()
    return v if v else default


def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except:
        return default


def env_float(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except:
        return default


# =========================
# CONFIG
# =========================

AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "") or "").rstrip("/")

PREFIX_PRIORITY = (env_nonempty("B_PREFIX_PRIORITY", "ugc/final_priority") or "ugc/final_priority").strip().strip("/")
PREFIX_MANUAL = (env_nonempty("B_PREFIX_MANUAL", "ugc/final_manual") or "ugc/final_manual").strip().strip("/")
PREFIX_AUTO = (env_nonempty("B_PREFIX_AUTO", "ugc/final_clean") or "ugc/final_clean").strip().strip("/")

META_FINAL_PREFIX = (env_nonempty("B_META_FINAL_PREFIX", "ugc/meta/final_clean") or "ugc/meta/final_clean").strip().strip("/")

STATE_KEY = env_nonempty("B_STATE_KEY", "ugc/state/mode_b_state.json")

B_MAX_PUBLISH_PER_RUN = env_int("B_MAX_PUBLISH_PER_RUN", 2)

B_ONLY_KEYS_CONTAIN = env_nonempty("B_ONLY_KEYS_CONTAIN", "")
B_AVOID_SAME_SOURCE_PER_RUN = env_bool("B_AVOID_SAME_SOURCE_PER_RUN", True)

ENABLE_INSTAGRAM = env_bool("ENABLE_INSTAGRAM", False)
ENABLE_FACEBOOK = env_bool("ENABLE_FACEBOOK", False)
ENABLE_TIKTOK = env_bool("ENABLE_TIKTOK", False)
ENABLE_SHORTS = env_bool("ENABLE_SHORTS", False)

DRY_RUN = env_bool("DRY_RUN", False)

GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

IG_USER_ID = env_nonempty("IG_USER_ID")
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")

FB_PAGE_ID = env_nonempty("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = env_nonempty("FB_PAGE_ACCESS_TOKEN")


# =========================
# R2 CLIENT
# =========================

def r2():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def list_keys(prefix):
    s3 = r2()
    items = []
    token = None

    while True:
        params = {
            "Bucket": BUCKET_NAME,
            "Prefix": f"{prefix}/",
            "MaxKeys": 200,
        }

        if token:
            params["ContinuationToken"] = token

        resp = s3.list_objects_v2(**params)

        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if not k.endswith("/") and k.lower().endswith(".mp4"):
                items.append(
                    {
                        "key": k,
                        "last_modified": obj.get("LastModified"),
                    }
                )

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    items.sort(
        key=lambda x: x["last_modified"] or 0,
        reverse=True,
    )

    return [x["key"] for x in items]


# =========================
# STATE
# =========================

def load_state():
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=STATE_KEY)
        st = json.loads(obj["Body"].read())
    except:
        st = {}

    if "published" not in st:
        st["published"] = []

    return st


def save_state(st):
    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=STATE_KEY,
        Body=json.dumps(st).encode(),
        ContentType="application/json",
    )


# =========================
# SOURCE GROUP
# =========================

def extract_source_group(key):
    """
    Agrupa reels hermanos del mismo lote.
    """

    base = os.path.basename(key).rsplit(".", 1)[0]

    if "__hype__" in base:
        base = base.split("__hype__", 1)[0]

    m = re.match(r"^(\d{8}_\d{6})_\d+$", base)
    if m:
        return m.group(1)

    parts = base.split("__")

    if len(parts) >= 3:
        return "__".join(parts[:-2])

    return base


# =========================
# DIVERSIFY QUEUE
# =========================

def diversify_queue(queue):
    if not B_AVOID_SAME_SOURCE_PER_RUN:
        return queue

    groups = {}
    order = []

    for key in queue:
        group = extract_source_group(key)

        if group not in groups:
            groups[group] = []
            order.append(group)

        groups[group].append(key)

    diversified = []

    while True:
        added = 0

        for group in order:
            if groups[group]:
                diversified.append(groups[group].pop(0))
                added += 1

        if added == 0:
            break

    return diversified


# =========================
# CAPTION
# =========================

def build_caption(key):

    title = os.path.basename(key).replace(".mp4", "")

    return f"""🎮 Gaming moment

{title}

#gaming #esports #reels
"""


# =========================
# META GRAPH HELPERS
# =========================

def ig_publish(video_url, caption):

    print("IG publish: creando container...")

    resp = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": IG_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )

    container = resp.json()["id"]

    print("IG publish: esperando container...", container)

    while True:

        status = requests.get(
            f"{GRAPH_BASE}/{container}",
            params={
                "fields": "status_code",
                "access_token": IG_ACCESS_TOKEN,
            },
        ).json()["status_code"]

        print("IG status:", status)

        if status == "FINISHED":
            break

        time.sleep(3)

    print("IG publish: publicando...")

    r = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media_publish",
        data={
            "creation_id": container,
            "access_token": IG_ACCESS_TOKEN,
        },
    )

    return r.json()


def fb_publish(video_url, caption):

    print("FB reel upload START")

    start = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "START",
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
    ).json()

    upload_url = start["upload_url"]
    video_id = start["video_id"]

    print("FB reel upload TRANSFER")

    transfer = requests.post(
        upload_url,
        headers={
            "Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}",
            "file_url": video_url,
        },
    )

    print("FB transfer:", transfer.json())

    print("FB reel upload FINISH")

    finish = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "FINISH",
            "video_id": video_id,
            "video_state": "PUBLISHED",
            "description": caption,
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
    )

    return finish.json()


# =========================
# PUBLISH
# =========================

def publish(key):

    public_url = f"{R2_PUBLIC_BASE_URL}/{key}"
    caption = build_caption(key)

    print("PUBLICANDO VIDEO:")
    print("KEY:", key)
    print("URL:", public_url)

    if ENABLE_INSTAGRAM:
        print("→ Publicando en Instagram...")
        ig = ig_publish(public_url, caption)
        print("IG OK:", ig)

    if ENABLE_FACEBOOK:
        print("→ Publicando en Facebook...")
        fb = fb_publish(public_url, caption)
        print("FB OK:", fb)

    print("Publicado OK\n")


# =========================
# MAIN
# =========================

def run_mode_b():

    print("===== MODE B (PUBLISHER) START =====")
    print("MODE B VERSION: REAL_PUBLISH_DIVERSIFIED_V3")
    print("B_MAX_PUBLISH_PER_RUN:", B_MAX_PUBLISH_PER_RUN)
    print("B_AVOID_SAME_SOURCE_PER_RUN:", B_AVOID_SAME_SOURCE_PER_RUN)

    state = load_state()
    published = set(state["published"])

    priority = list_keys(PREFIX_PRIORITY)
    manual = list_keys(PREFIX_MANUAL)
    auto = list_keys(PREFIX_AUTO)

    print("Priority:", len(priority))
    print("Manual:", len(manual))
    print("Auto:", len(auto))

    queue = []
    queue.extend(priority)
    queue.extend(manual)
    queue.extend(auto)

    if B_ONLY_KEYS_CONTAIN:
        queue = [k for k in queue if B_ONLY_KEYS_CONTAIN in k]
        print("Filtro B_ONLY_KEYS_CONTAIN:", B_ONLY_KEYS_CONTAIN)
        print("Queue tras filtro:", len(queue))

    queue = diversify_queue(queue)

    print("Queue final diversificada:", len(queue))

    count = 0

    for key in queue:

        if count >= B_MAX_PUBLISH_PER_RUN:
            break

        if key in published:
            continue

        print("Procesando:", key)
        print("SOURCE GROUP:", extract_source_group(key))

        try:
            publish(key)
            published.add(key)
            count += 1

        except Exception as e:
            print("ERROR publicando:", e)

    state["published"] = list(published)[-5000:]
    save_state(state)

    print("Publicados en esta corrida:", count)
    print("===== MODE B DONE =====")
