# ugc_mode_g.py
# TREND HUNTER ENGINE

import os
import requests
import random

from ugc_mode_b import openai_text


REDDIT_URL = "https://www.reddit.com/r/esports/hot.json"

HEADERS = {
    "User-Agent": "trend-bot"
}


def get_reddit_trends():

    r = requests.get(REDDIT_URL, headers=HEADERS)

    data = r.json()

    posts = data["data"]["children"]

    trends = []

    for p in posts:

        title = p["data"]["title"]

        score = p["data"]["score"]

        if score > 500:

            trends.append(title)

    return trends


def generate_content_idea(trend):

    prompt = f"""
Eres editor viral de esports.

Tema tendencia:

{trend}

Genera:

1 idea de video viral
1 caption corto polémico
1 hook fuerte

Formato:

HOOK:
IDEA:
CAPTION:
"""

    return openai_text(prompt)


def run_mode_g():

    print("===== MODE G TREND HUNTER =====")

    trends = get_reddit_trends()

    if not trends:

        print("No trends found")

        return

    trend = random.choice(trends)

    print("Trend detectado:", trend)

    idea = generate_content_idea(trend)

    print("Contenido generado:\n")

    print(idea)

    print("===== MODE G DONE =====")
