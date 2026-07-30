#!/usr/bin/env python3
"""Publish the explicitly approved Wolverine V4 to four platforms once."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import boto3
import requests
from botocore.exceptions import ClientError


VIDEO = Path("artifacts/approved/wolverine-v4/wolverine-v4-approved.mp4")
EXPECTED_SHA256 = "6748e3a4f042bfe1e89429794dbc45db3ea93673e03fd823ca0df7c33ee14ec8"
APPROVAL_TOKEN = "wolverine-v4-multiplatform-approved-2026-07-30-20h-colombia"
R2_VIDEO_KEY = f"approved/multiplatform/wolverine-v4-{EXPECTED_SHA256[:16]}.mp4"
RECEIPT = Path("artifacts/wolverine-v4-multiplatform-publication-result.json")

CAPTION = """Marvel’s Wolverine mostró su historia: Jean Grey, Deathstrike y La Mano entran en guerra.

¿Superará a Marvel’s Spider-Man?

Llega el 15 de septiembre de 2026 a PS5.

Fuente audiovisual: PlayStation / Insomniac Games.

#MarvelsWolverine #Wolverine #PlayStation5 #PS5 #GamingLatam"""

YT_TITLE = "Wolverine mostró su HISTORIA ¿superará a Spider-Man? #Shorts"
YT_DESCRIPTION = """Jean Grey, Deathstrike y La Mano entran en guerra en Marvel’s Wolverine.

Lanzamiento: 15 de septiembre de 2026 en PS5.

Fuente audiovisual: PlayStation / Insomniac Games.
Fuente editorial:
https://blog.playstation.com/2026/07/23/marvels-wolverine-story-trailer-new-art-composer-details-and-more/

#MarvelsWolverine #Wolverine #PS5 #Shorts #GamingLatam"""

THREADS_TEXT = """Wolverine ya mostró su historia: Jean Grey, Deathstrike y La Mano entran en guerra.

¿Superará a Spider-Man?

Fuente audiovisual: PlayStation / Insomniac Games."""


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
    actual = hashlib.sha256(VIDEO.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError("El video no coincide con el archivo aprobado")


def write_receipt(payload: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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


def graph_get(base: str, path: str, token: str, **params: str) -> dict[str, Any]:
    response = requests.get(
        f"{base}/{path.lstrip('/')}",
        params={**params, "access_token": token},
        timeout=60,
    )
    raise_platform_error(response, f"GET {path}")
    return response.json()


def graph_post(base: str, path: str, token: str, **data: str) -> dict[str, Any]:
    response = requests.post(
        f"{base}/{path.lstrip('/')}",
        data={**data, "access_token": token},
        timeout=120,
    )
    raise_platform_error(response, f"POST {path}")
    return response.json()


def wait_meta_container(
    base: str,
    creation_id: str,
    token: str,
    *,
    status_field: str,
) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        payload = graph_get(
            base,
            creation_id,
            token,
            fields=f"{status_field},status,error_message",
        )
        status = str(
            payload.get(status_field) or payload.get("status") or ""
        ).upper()
        if status in {"FINISHED", "PUBLISHED"}:
            return
        if status in {"ERROR", "FAILED", "EXPIRED"}:
            raise RuntimeError(f"Meta no procesó el video: {payload}")
        time.sleep(5)
    raise TimeoutError("Meta no terminó de procesar el video")


def publish_instagram(
    r2: Any,
    bucket: str,
    video_url: str,
    graph_base: str,
) -> dict[str, Any]:
    key = "publication-ledger/instagram/wolverine-v4.json"
    existing = ledger_get(r2, bucket, key)
    if existing and existing.get("status") == "published":
        return existing
    user_id = required("IG_USER_ID")
    token = required("IG_ACCESS_TOKEN")
    container = graph_post(
        graph_base,
        f"{user_id}/media",
        token,
        media_type="REELS",
        video_url=video_url,
        caption=CAPTION,
        share_to_feed="true",
    )
    creation_id = str(container["id"])
    wait_meta_container(
        graph_base,
        creation_id,
        token,
        status_field="status_code",
    )
    published = graph_post(
        graph_base,
        f"{user_id}/media_publish",
        token,
        creation_id=creation_id,
    )
    media_id = str(published["id"])
    details = graph_get(
        graph_base,
        media_id,
        token,
        fields="id,media_type,permalink,timestamp",
    )
    payload = {
        "status": "published",
        "platform": "instagram",
        "media_id": media_id,
        "permalink": details.get("permalink"),
        "verified_at": now(),
    }
    ledger_put(r2, bucket, key, payload)
    return payload


def publish_facebook(
    r2: Any,
    bucket: str,
    video_url: str,
    graph_base: str,
) -> dict[str, Any]:
    key = "publication-ledger/facebook/wolverine-v4.json"
    existing = ledger_get(r2, bucket, key)
    if existing and existing.get("status") == "published":
        return existing
    page_id = required("FB_PAGE_ID")
    token = required("FB_PAGE_ACCESS_TOKEN")
    start = graph_post(
        graph_base,
        f"{page_id}/video_reels",
        token,
        upload_phase="start",
    )
    upload = requests.post(
        start["upload_url"],
        headers={"Authorization": f"OAuth {token}", "file_url": video_url},
        timeout=180,
    )
    raise_platform_error(upload, "Facebook Reels UPLOAD")
    finish = graph_post(
        graph_base,
        f"{page_id}/video_reels",
        token,
        upload_phase="finish",
        video_id=str(start["video_id"]),
        video_state="PUBLISHED",
        description=CAPTION,
    )
    payload = {
        "status": "published",
        "platform": "facebook",
        "video_id": str(start["video_id"]),
        "response": finish,
        "permalink": f"https://www.facebook.com/{start['video_id']}",
        "verified_at": now(),
    }
    ledger_put(r2, bucket, key, payload)
    return payload


def publish_youtube(r2: Any, bucket: str) -> dict[str, Any]:
    key = "publication-ledger/youtube/wolverine-v4.json"
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
                "tags": [
                    "Marvel's Wolverine",
                    "Wolverine",
                    "PS5",
                    "gaming",
                    "shorts",
                ],
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(
            str(VIDEO),
            mimetype="video/mp4",
            resumable=True,
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


def publish_threads(
    r2: Any,
    bucket: str,
    video_url: str,
) -> dict[str, Any]:
    key = "publication-ledger/threads/wolverine-v4.json"
    existing = ledger_get(r2, bucket, key)
    if existing and existing.get("status") == "published":
        return existing
    user_id = required("THREADS_USER_ID")
    token = required("THREADS_USER_ACCESS_TOKEN")
    base = "https://graph.threads.net/v1.0"
    container = graph_post(
        base,
        f"{user_id}/threads",
        token,
        media_type="VIDEO",
        video_url=video_url,
        text=THREADS_TEXT,
    )
    creation_id = str(container["id"])
    wait_meta_container(
        base,
        creation_id,
        token,
        status_field="status",
    )
    published = graph_post(
        base,
        f"{user_id}/threads_publish",
        token,
        creation_id=creation_id,
    )
    media_id = str(published["id"])
    details = graph_get(
        base,
        media_id,
        token,
        fields="id,media_type,permalink,timestamp",
    )
    payload = {
        "status": "published",
        "platform": "threads",
        "media_id": media_id,
        "permalink": details.get("permalink"),
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
    public_check = requests.head(video_url, timeout=30, allow_redirects=True)
    public_check.raise_for_status()
    graph_version = os.environ.get("GRAPH_VERSION", "v25.0")
    graph_base = f"https://graph.facebook.com/{graph_version}"
    publishers: dict[str, Callable[[], dict[str, Any]]] = {
        "instagram": lambda: publish_instagram(
            r2,
            bucket,
            video_url,
            graph_base,
        ),
        "facebook": lambda: publish_facebook(
            r2,
            bucket,
            video_url,
            graph_base,
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
