#!/usr/bin/env python3
"""Publish the approved Dragon Ball video to Facebook, YouTube and Threads."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import requests
from botocore.exceptions import ClientError


VIDEO = Path(
    "artifacts/approved/dragonball-batch-v1/dragonball-master-approved.mp4"
)
EXPECTED_SHA256 = "4cb7f4d63a40dcee709cf9ff34a82faddfd9300d1c8a61d5892765f328bd5044"
APPROVAL_TOKEN = "dragonball-v1-multiplatform-approved-2026-07-30"
R2_VIDEO_KEY = f"approved/multiplatform/dragonball-v1-{EXPECTED_SHA256[:16]}.mp4"
RECEIPT = Path("artifacts/dragonball-v1-multiplatform-publication-result.json")

FB_CAPTION = """Dragon Ball: Sparking! ZERO amplía su universo con más de 30 luchadores, cuatro escenarios y un nuevo modo individual.

¿Vale la pena volver? Te leemos.

Fuente audiovisual: Bandai Namco Entertainment.

#DragonBallSparkingZero #DragonBall #GamingLatam #Videojuegos"""

YT_TITLE = "Sparking Zero suma MÁS DE 30 personajes #Shorts"
YT_DESCRIPTION = """Nuevos luchadores, cuatro escenarios y un modo individual llegan a Dragon Ball: Sparking! ZERO.

¿Cuál vas a entrenar primero?

Fuente audiovisual: Bandai Namco Entertainment.
https://www.youtube.com/watch?v=SZhl5-ag0tA

#DragonBall #SparkingZero #Shorts #GamingLatam"""

THREADS_TEXT = """Sparking Zero ya supera los 200 personajes y ahora suma más de 30.

¿Esto mejora el juego o ya son demasiados?

Fuente audiovisual: Bandai Namco Entertainment."""


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la configuración protegida: {name}")
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate() -> None:
    if os.environ.get("PUBLICATION_APPROVAL_TOKEN") != APPROVAL_TOKEN:
        raise RuntimeError("Falta la autorización exacta de esta publicación")
    if not VIDEO.is_file():
        raise RuntimeError("No existe el video aprobado")
    if hashlib.sha256(VIDEO.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("El video no coincide con el archivo aprobado")


def write_receipt(payload: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def raise_platform_error(response: requests.Response, label: str) -> None:
    if response.ok:
        return
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        error = {}
    details = {
        "status": response.status_code,
        "code": error.get("code"),
        "subcode": error.get("error_subcode"),
        "type": error.get("type"),
        "message": error.get("message") or "respuesta no detallada",
    }
    raise RuntimeError(f"{label}: {json.dumps(details, ensure_ascii=False)}")


def ledger_get(r2: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = r2.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except r2.exceptions.NoSuchKey:
        return None
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise


def ledger_put(r2: Any, bucket: str, key: str, payload: dict[str, Any]) -> None:
    r2.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def publish_facebook(
    r2: Any, bucket: str, video_url: str, graph_base: str
) -> dict[str, Any]:
    key = "publication-ledger/facebook/dragonball-v1.json"
    existing = ledger_get(r2, bucket, key)
    if existing and existing.get("status") == "published":
        return existing

    page_id = required("FB_PAGE_ID")
    token = required("FB_PAGE_ACCESS_TOKEN")
    accounts = requests.get(
        f"{graph_base}/me/accounts",
        params={"fields": "id,access_token", "access_token": token},
        timeout=60,
    )
    if accounts.ok:
        matches = [
            item
            for item in accounts.json().get("data", [])
            if str(item.get("id")) == page_id
        ]
        if matches:
            token = str(matches[0].get("access_token") or token)
    start = requests.post(
        f"{graph_base}/{page_id}/video_reels",
        data={"upload_phase": "start", "access_token": token},
        timeout=60,
    )
    raise_platform_error(start, "Facebook Reels START")
    started = start.json()
    upload = requests.post(
        started["upload_url"],
        headers={"Authorization": f"OAuth {token}", "file_url": video_url},
        timeout=120,
    )
    raise_platform_error(upload, "Facebook Reels UPLOAD")
    finish = requests.post(
        f"{graph_base}/{page_id}/video_reels",
        data={
            "upload_phase": "finish",
            "video_id": started["video_id"],
            "video_state": "PUBLISHED",
            "description": FB_CAPTION,
            "access_token": token,
        },
        timeout=60,
    )
    raise_platform_error(finish, "Facebook Reels FINISH")
    payload = {
        "status": "published",
        "platform": "facebook",
        "video_id": started["video_id"],
        "response": finish.json(),
        "permalink": f"https://www.facebook.com/{started['video_id']}",
        "verified_at": now(),
    }
    ledger_put(r2, bucket, key, payload)
    return payload


def publish_youtube(r2: Any, bucket: str) -> dict[str, Any]:
    key = "publication-ledger/youtube/dragonball-v1.json"
    existing = ledger_get(r2, bucket, key)
    if existing and existing.get("status") == "published":
        return existing

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    credentials = Credentials(
        None,
        refresh_token=required("YOUTUBE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=required("YOUTUBE_CLIENT_ID"),
        client_secret=required("YOUTUBE_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    youtube = build("youtube", "v3", credentials=credentials)
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": YT_TITLE,
                "description": YT_DESCRIPTION,
                "categoryId": "20",
                "tags": ["Dragon Ball", "Sparking Zero", "gaming", "shorts"],
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(
            str(VIDEO), mimetype="video/mp4", resumable=True
        ),
    )
    response = None
    while response is None:
        _, response = request.next_chunk()
    video_id = str(response["id"])
    payload = {
        "status": "published",
        "platform": "youtube",
        "video_id": video_id,
        "permalink": f"https://www.youtube.com/watch?v={video_id}",
        "verified_at": now(),
    }
    ledger_put(r2, bucket, key, payload)
    return payload


def threads_post(
    base: str, path: str, token: str, data: dict[str, str]
) -> dict[str, Any]:
    response = requests.post(
        f"{base}/{path.lstrip('/')}",
        data={**data, "access_token": token},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def publish_threads(
    r2: Any, bucket: str, video_url: str
) -> dict[str, Any]:
    key = "publication-ledger/threads/dragonball-v1.json"
    existing = ledger_get(r2, bucket, key)
    if existing and existing.get("status") == "published":
        return existing

    user_id = required("THREADS_USER_ID")
    token = required("THREADS_USER_ACCESS_TOKEN")
    base = "https://graph.threads.net/v1.0"
    container = threads_post(
        base,
        f"{user_id}/threads",
        token,
        {"media_type": "VIDEO", "video_url": video_url, "text": THREADS_TEXT},
    )
    creation_id = str(container["id"])
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        response = requests.get(
            f"{base}/{creation_id}",
            params={"fields": "status,error_message", "access_token": token},
            timeout=60,
        )
        response.raise_for_status()
        status_payload = response.json()
        status = str(status_payload.get("status", "")).upper()
        if status in {"FINISHED", "PUBLISHED"}:
            break
        if status in {"ERROR", "FAILED", "EXPIRED"}:
            raise RuntimeError(f"Threads no procesó el video: {status_payload}")
        time.sleep(5)
    else:
        raise TimeoutError("Threads no terminó de procesar el video")

    published = threads_post(
        base,
        f"{user_id}/threads_publish",
        token,
        {"creation_id": creation_id},
    )
    media_id = str(published["id"])
    details = requests.get(
        f"{base}/{media_id}",
        params={
            "fields": "id,media_type,permalink,timestamp",
            "access_token": token,
        },
        timeout=60,
    )
    details.raise_for_status()
    media = details.json()
    payload = {
        "status": "published",
        "platform": "threads",
        "media_id": media_id,
        "permalink": media.get("permalink"),
        "verified_at": now(),
    }
    ledger_put(r2, bucket, key, payload)
    return payload


def main() -> None:
    validate()
    endpoint = required("R2_ENDPOINT_URL")
    bucket = required("BUCKET_NAME")
    public_base = required("R2_PUBLIC_BASE_URL").rstrip("/")
    r2 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=required("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=required("AWS_SECRET_ACCESS_KEY"),
        region_name="auto",
    )
    r2.put_object(
        Bucket=bucket,
        Key=R2_VIDEO_KEY,
        Body=VIDEO.read_bytes(),
        ContentType="video/mp4",
    )
    video_url = f"{public_base}/{R2_VIDEO_KEY}"
    check = requests.head(video_url, timeout=30, allow_redirects=True)
    check.raise_for_status()
    graph_version = os.environ.get("GRAPH_VERSION", "v25.0")
    publishers = {
        "facebook": lambda: publish_facebook(
            r2, bucket, video_url, f"https://graph.facebook.com/{graph_version}"
        ),
        "youtube": lambda: publish_youtube(r2, bucket),
        "threads": lambda: publish_threads(r2, bucket, video_url),
    }
    result: dict[str, Any] = {}
    for platform, publisher in publishers.items():
        try:
            result[platform] = publisher()
        except Exception as exc:
            result[platform] = {
                "status": "blocked",
                "platform": platform,
                "reason": str(exc),
                "verified_at": now(),
            }
    write_receipt(result)
    print(json.dumps(result, ensure_ascii=False))
    if any(item.get("status") != "published" for item in result.values()):
        raise RuntimeError("Una o más plataformas quedaron bloqueadas")


if __name__ == "__main__":
    main()
