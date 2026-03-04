# ugc_mode_f.py
# VIRAL AI BRAIN

import os
import requests
from datetime import datetime

from ugc_mode_b import (
    r2_client,
    s3_get_json,
    s3_put_json,
    r2_public_url,
    ig_publish_reel
)

GRAPH_VERSION = os.getenv("GRAPH_VERSION","v25.0")
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")
IG_USER_ID = os.getenv("IG_USER_ID")

GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"

VIRAL_THRESHOLD = int(os.getenv("VIRAL_THRESHOLD", "5000"))

STATE_KEY = "ugc/state/viral_state.json"


# ------------------------------------
# load state
# ------------------------------------

def load_state():

    st = s3_get_json(STATE_KEY)

    if not st:

        st = {"reposted":[]}

    return st


def save_state(st):

    s3_put_json(STATE_KEY, st)


# ------------------------------------
# get reels
# ------------------------------------

def get_recent_media():

    url = f"{GRAPH_BASE}/{IG_USER_ID}/media"

    params = {
        "fields":"id,caption,media_type,permalink",
        "access_token":IG_ACCESS_TOKEN
    }

    r = requests.get(url,params=params)

    return r.json().get("data",[])


# ------------------------------------
# get insights
# ------------------------------------

def get_insights(media_id):

    url = f"{GRAPH_BASE}/{media_id}/insights"

    params = {
        "metric":"plays,likes,comments",
        "access_token":IG_ACCESS_TOKEN
    }

    r = requests.get(url,params=params)

    data = r.json().get("data",[])

    metrics = {}

    for d in data:

        metrics[d["name"]] = d["values"][0]["value"]

    return metrics


# ------------------------------------
# MAIN
# ------------------------------------

def run_mode_f():

    print("===== MODE F VIRAL BRAIN =====")

    state = load_state()

    media = get_recent_media()

    for m in media:

        if m["media_type"] != "VIDEO":

            continue

        media_id = m["id"]

        if media_id in state["reposted"]:

            continue

        insights = get_insights(media_id)

        plays = insights.get("plays",0)

        print("Reel:",media_id,"plays:",plays)

        if plays > VIRAL_THRESHOLD:

            caption = m["caption"] + "\n\n🔥 este clip volvió por demanda"

            permalink = m["permalink"]

            ig_publish_reel(permalink, caption)

            state["reposted"].append(media_id)

            print("Repost realizado")

    save_state(state)

    print("===== MODE F DONE =====")
