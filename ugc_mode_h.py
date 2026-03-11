import os
import re
import json
import random
import hashlib
import subprocess
import tempfile
from datetime import datetime, timezone

import boto3

def env(name, default=None):
    v = os.getenv(name)
    if not v:
        return default
    return v.strip()

AWS_ACCESS_KEY_ID = env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env("R2_ENDPOINT_URL")
BUCKET_NAME = env("BUCKET_NAME")

INPUT_PREFIX = env("MODE_H_INPUT_PREFIX", "ugc/library/clips")
OUTPUT_PREFIX = env("MODE_H_OUTPUT_PREFIX", "ugc/final_clean")
META_PREFIX = env("MODE_H_META_PREFIX", "ugc/meta/final_clean")
STATE_KEY = env("MODE_H_STATE_KEY", "ugc/state/mode_h_state.json")

MAX_ITEMS = int(env("MODE_H_MAX_ITEMS", "6"))

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
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    for obj in resp.get("Contents", []):
        k = obj["Key"]
        if k.endswith(".mp4"):
            keys.append((k, obj["LastModified"]))
    keys.sort(key=lambda x: x[1], reverse=True)
    return [k[0] for k in keys]

def load_state():
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=STATE_KEY)
        return json.loads(obj["Body"].read())
    except:
        return {"processed": []}

def save_state(st):
    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=STATE_KEY,
        Body=json.dumps(st).encode(),
        ContentType="application/json",
    )

def detect_game(key):
    k = key.lower()

    if "valorant" in k:
        return "Valorant"
    if "fortnite" in k:
        return "Fortnite"
    if "warzone" in k:
        return "Warzone"
    if "apex" in k:
        return "Apex"
    if "minecraft" in k:
        return "Minecraft"
    if "cs2" in k:
        return "CS2"
    if "league" in k:
        return "LeagueOfLegends"
    if "fc" in k:
        return "EASportsFC"
    if "f1" in k:
        return "F1"

    return "Esports"

def process_clip(key):
    game = detect_game(key)

    base = os.path.basename(key).replace(".mp4","")
    h = hashlib.sha1(base.encode()).hexdigest()[:10]

    final_key = f"{OUTPUT_PREFIX}/{base}__hype__{h}.mp4"
    meta_key = f"{META_PREFIX}/{base}__hype__{h}.json"

    s3 = r2()

    tmp = tempfile.NamedTemporaryFile(delete=False)
    s3.download_file(BUCKET_NAME, key, tmp.name)

    # solo re-sube el mismo video (packer simple)
    s3.upload_file(tmp.name, BUCKET_NAME, final_key)

    meta = {
        "source_clip": key,
        "game": game,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=meta_key,
        Body=json.dumps(meta).encode(),
        ContentType="application/json",
    )

    print("FINAL:", final_key)
    print("GAME:", game)

def run():
    print("===== MODE H PRO =====")

    st = load_state()
    done = set(st["processed"])

    clips = list_keys(INPUT_PREFIX)

    count = 0

    for key in clips:
        if count >= MAX_ITEMS:
            break

        if key in done:
            continue

        process_clip(key)

        done.add(key)
        count += 1

    st["processed"] = list(done)[-5000:]
    save_state(st)

    print("DONE:", count)

if __name__ == "__main__":
    run()
