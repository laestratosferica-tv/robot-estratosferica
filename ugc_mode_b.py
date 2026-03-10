import os
import json
import time

import boto3
import requests


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
PREFIX_AUTO = (env_nonempty("B_PREFIX_AUTO", "ugc/final") or "ugc/final").strip().strip("/")

STATE_KEY = env_nonempty("B_STATE_KEY", "ugc/state/mode_b_state.json")
B_MAX_PUBLISH_PER_RUN = env_int("B_MAX_PUBLISH_PER_RUN", 2)

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

META_FINAL_PREFIX = (env_nonempty("B_META_FINAL_PREFIX", "ugc/meta/final") or "ugc/meta/final").strip().strip("/")


# =========================
# R2
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
    keys = []
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
                keys.append(k)

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    keys.sort()
    return keys


def get_public_url(key):
    if not R2_PUBLIC_BASE_URL:
        raise RuntimeError("Falta R2_PUBLIC_BASE_URL")
    return f"{R2_PUBLIC_BASE_URL}/{key}"


def load_meta_for_video(key):
    """
    Busca meta de H:
    ugc/final/foo__hype__abc.mp4
    -> ugc/meta/final/foo__hype__abc.json
    """
    base = os.path.basename(key).rsplit(".", 1)[0] + ".json"
    meta_key = f"{META_FINAL_PREFIX}/{base}"

    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=meta_key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


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
# CAPTIONS
# =========================

def build_caption_from_meta(key, meta):
    if not meta:
        title = os.path.basename(key).rsplit(".", 1)[0]
        return (
            "🎮 Clip gamer del día 🔥\n\n"
            f"{title}\n\n"
            "#gaming #esports #reels"
        )

    game_name = (meta.get("game_name") or "gaming").upper()
    hook = (meta.get("hook") or "ESTO FUE CINE").strip()
    cta = (meta.get("cta") or "¿TOP O HUMO?").strip()

    tags = {
        "VALORANT": "#valorant #gaming #esports",
        "CS2": "#cs2 #counterstrike #gaming",
        "LOL": "#leagueoflegends #lol #gaming",
        "FORTNITE": "#fortnite #gaming #clips",
        "WARZONE": "#warzone #cod #gaming",
        "APEX": "#apexlegends #gaming #esports",
        "MINECRAFT": "#minecraft #gaming #clips",
        "FC": "#easportsfc #fc25 #gaming",
        "F1": "#f1 #simracing #gaming",
        "GT": "#granturismo #simracing #gaming",
        "GAMER": "#gaming #esports #reels",
    }

    hashtag_line = tags.get(game_name, "#gaming #esports #reels")

    return (
        f"🎮 {game_name}\n"
        f"{hook}\n"
        f"{cta}\n\n"
        f"{hashtag_line}"
    )


# =========================
# META GRAPH HELPERS
# =========================

def raise_meta_error(resp, label):
    if resp.status_code >= 400:
        raise RuntimeError(f"{label} failed: {resp.status_code} {resp.text}")


def ig_api_post(path, data):
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.post(url, data=data, timeout=HTTP_TIMEOUT)
    raise_meta_error(r, f"IG POST {path}")
    return r.json()


def ig_api_get(path, params):
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    raise_meta_error(r, f"IG GET {path}")
    return r.json()


def ig_wait_container(creation_id, access_token, timeout_sec=900):
    start = time.time()

    while time.time() - start < timeout_sec:
        j = ig_api_get(
            f"{creation_id}",
            {"fields": "status_code", "access_token": access_token},
        )

        status = (j.get("status_code") or "").upper()
        print("IG status:", status)

        if status in ("FINISHED", "PUBLISHED"):
            return

        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {j}")

        time.sleep(3)

    raise TimeoutError(f"IG container not ready after {timeout_sec}s")


def ig_publish_reel(video_url, caption):
    if not (IG_USER_ID and IG_ACCESS_TOKEN):
        raise RuntimeError("Faltan IG_USER_ID o IG_ACCESS_TOKEN")

    print("IG publish: creando container...")
    j = ig_api_post(
        f"{IG_USER_ID}/media",
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": IG_ACCESS_TOKEN,
        },
    )

    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG reels create failed: {j}")

    print("IG publish: esperando container...", creation_id)
    ig_wait_container(creation_id, IG_ACCESS_TOKEN, timeout_sec=900)

    print("IG publish: publicando...")
    res = ig_api_post(
        f"{IG_USER_ID}/media_publish",
        {"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN},
    )
    return res


def fb_publish_reel(video_url, caption):
    if not (FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN):
        raise RuntimeError("Faltan FB_PAGE_ID o FB_PAGE_ACCESS_TOKEN")

    print("FB reel upload START")

    start_resp = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "START",
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )
    raise_meta_error(start_resp, "FB REELS START")
    start_json = start_resp.json()

    upload_url = start_json.get("upload_url")
    video_id = start_json.get("video_id")

    if not upload_url or not video_id:
        raise RuntimeError(f"FB START inválido: {start_json}")

    print("FB reel upload TRANSFER")
    transfer_resp = requests.post(
        upload_url,
        headers={
            "Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}",
            "file_url": video_url,
        },
        timeout=HTTP_TIMEOUT,
    )
    raise_meta_error(transfer_resp, "FB REELS TRANSFER")

    try:
        transfer_json = transfer_resp.json()
    except Exception:
        transfer_json = {"raw": transfer_resp.text}

    print("FB transfer:", transfer_json)

    print("FB reel upload FINISH")
    finish_resp = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "FINISH",
            "video_id": video_id,
            "video_state": "PUBLISHED",
            "description": caption[:2200],
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )
    raise_meta_error(finish_resp, "FB REELS FINISH")

    return finish_resp.json()


# =========================
# PUBLISH
# =========================

def publish_real(key):
    public_url = get_public_url(key)
    meta = load_meta_for_video(key)
    caption = build_caption_from_meta(key, meta)

    print("PUBLICANDO VIDEO:")
    print("KEY:", key)
    print("URL:", public_url)

    results = {
        "instagram": None,
        "facebook": None,
        "tiktok": None,
        "shorts": None,
    }

    if DRY_RUN:
        print("[DRY_RUN] No se publica realmente.")
        return {"ok": True, "dry_run": True, "results": results}

    if ENABLE_INSTAGRAM:
        print("→ Publicando en Instagram...")
        results["instagram"] = ig_publish_reel(public_url, caption)
        print("IG OK:", results["instagram"])

    if ENABLE_FACEBOOK:
        print("→ Publicando en Facebook...")
        results["facebook"] = fb_publish_reel(public_url, caption)
        print("FB OK:", results["facebook"])

    if ENABLE_TIKTOK:
        print("→ TikTok ENABLED pero aún no integrado en este B.")

    if ENABLE_SHORTS:
        print("→ YouTube Shorts ENABLED pero aún no integrado en este B.")

    print("Publicado OK\n")
    return {"ok": True, "dry_run": False, "results": results}


# =========================
# MAIN
# =========================

def run_mode_b():
    print("===== MODE B (PUBLISHER) START =====")
    print("B_MAX_PUBLISH_PER_RUN:", B_MAX_PUBLISH_PER_RUN)

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

    count = 0

    for key in queue:
        if count >= B_MAX_PUBLISH_PER_RUN:
            break

        if key in published:
            continue

        print("Procesando:", key)

        try:
            publish_real(key)
            published.add(key)
            count += 1
        except Exception as e:
            print("ERROR publicando:", key)
            print(str(e))
            # no lo marcamos como publicado si falló
            continue

    state["published"] = list(published)[-5000:]
    save_state(state)

    print("Publicados en esta corrida:", count)
    print("===== MODE B DONE =====")


if __name__ == "__main__":
    run_mode_b()
