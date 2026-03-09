import os
import json
from datetime import datetime, timezone

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

ENABLE_INSTAGRAM = env_bool("ENABLE_INSTAGRAM", False)
ENABLE_FACEBOOK = env_bool("ENABLE_FACEBOOK", False)
ENABLE_TIKTOK = env_bool("ENABLE_TIKTOK", False)
ENABLE_SHORTS = env_bool("ENABLE_SHORTS", False)


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
        params = {"Bucket": BUCKET_NAME, "Prefix": f"{prefix}/", "MaxKeys": 200}
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
# PUBLISH
# =========================

def publish_stub(key):
    """
    Aquí luego se conectará IG / FB / TikTok / Shorts.
    Por ahora solo log.
    """

    public_url = f"{R2_PUBLIC_BASE_URL}/{key}" if R2_PUBLIC_BASE_URL else key

    print("PUBLICANDO VIDEO:")
    print("KEY:", key)
    print("URL:", public_url)

    if ENABLE_INSTAGRAM:
        print("→ Instagram ENABLED (pendiente integración)")

    if ENABLE_FACEBOOK:
        print("→ Facebook ENABLED (pendiente integración)")

    if ENABLE_TIKTOK:
        print("→ TikTok ENABLED (pendiente integración)")

    if ENABLE_SHORTS:
        print("→ YouTube Shorts ENABLED (pendiente integración)")

    print("Publicado OK\n")


# =========================
# MAIN
# =========================

def run_mode_b():

    print("===== MODE B (PUBLISHER) START =====")

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

        if key in published:
            continue

        print("Procesando:", key)

        publish_stub(key)

        published.add(key)
        count += 1

    state["published"] = list(published)[-5000:]
    save_state(state)

    print("Publicados en esta corrida:", count)
    print("===== MODE B DONE =====")


if __name__ == "__main__":
    run_mode_b()
