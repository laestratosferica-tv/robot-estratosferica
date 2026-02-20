import os
import json
import requests
import feedparser
import boto3
from datetime import datetime
from dateutil import parser

# ==============================
# CONFIGURACIÓN
# ==============================

BUCKET_NAME = os.environ["BUCKET_NAME"]
R2_ENDPOINT = os.environ["R2_ENDPOINT_URL"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")

RSS_FEEDS = [
    "https://www.espn.com/espn/rss/news",
    "https://www.marca.com/rss/futbol/primera-division.xml"
]

# ==============================
# CONEXIÓN R2
# ==============================

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)


# ==============================
# FUNCIONES
# ==============================

def get_articles():
    articles = []

    for url in RSS_FEEDS:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            try:
                published = parser.parse(entry.published)
            except:
                published = datetime.utcnow()

            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": published.isoformat()
            })

    return articles


def save_to_r2(data):
    filename = f"articles_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=filename,
        Body=json.dumps(data, ensure_ascii=False),
        ContentType="application/json"
    )

    print("Archivo guardado en R2:", filename)


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("Obteniendo artículos...")
    articles = get_articles()
    print(f"{len(articles)} artículos encontrados")

    save_to_r2(articles)
