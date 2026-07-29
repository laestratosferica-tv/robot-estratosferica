#!/usr/bin/env python3
"""Publish one explicitly approved Instagram Reel and write an audit receipt."""

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
    "artifacts/review/halo-multiverse-v1/"
    "halo-multiverse-pilot-v9-public-candidate.mp4"
)
EXPECTED_SHA256 = "c33a338b3ed1c5450cc4cfb7ccb1291d17280817b44061fa11077c677c83b917"
APPROVAL_TOKEN = "halo-v9-instagram-approved-2026-07-29"
R2_VIDEO_KEY = f"approved/instagram/halo-v9-{EXPECTED_SHA256[:16]}.mp4"
R2_LEDGER_KEY = "publication-ledger/instagram/halo-v9.json"
RECEIPT = Path("artifacts/instagram-halo-v9-publication-result.json")
CAPTION = """Halo cambió antes de empezar. 👀

Tres misiones nuevas ponen al Jefe Maestro y a Johnson tras las líneas enemigas. ¿Entras?

Fuente audiovisual: tráiler oficial de HALO. Fragmentos breves transformados con narración y edición de La Estratosférica; audio original eliminado.

Fuente: https://www.youtube.com/watch?v=G4sUx2nX5EQ

#Halo #HaloCampaignEvolved #Gaming #Videojuegos #LaEstratosférica"""


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la configuración protegida: {name}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_receipt(payload: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_video() -> None:
    if not VIDEO.is_file():
        raise RuntimeError(f"No existe el video aprobado: {VIDEO}")
    actual_sha = hashlib.sha256(VIDEO.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError("El video no coincide con el archivo aprobado")


def graph_post(
    graph_base: str,
    path: str,
    data: dict[str, str],
) -> dict[str, Any]:
    response = requests.post(
        f"{graph_base}/{path.lstrip('/')}",
        data=data,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Instagram rechazó la operación: {payload['error']}")
    return payload


def graph_get(
    graph_base: str,
    path: str,
    params: dict[str, str],
) -> dict[str, Any]:
    response = requests.get(
        f"{graph_base}/{path.lstrip('/')}",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Instagram rechazó la consulta: {payload['error']}")
    return payload


def main() -> None:
    if os.environ.get("PUBLICATION_APPROVAL_TOKEN") != APPROVAL_TOKEN:
        raise RuntimeError("Falta la autorización exacta de esta publicación")
    validate_video()

    endpoint = required_env("R2_ENDPOINT_URL")
    bucket = required_env("BUCKET_NAME")
    public_base = required_env("R2_PUBLIC_BASE_URL").rstrip("/")
    access_key = required_env("AWS_ACCESS_KEY_ID")
    secret_key = required_env("AWS_SECRET_ACCESS_KEY")
    ig_user_id = required_env("IG_USER_ID")
    ig_token = required_env("IG_ACCESS_TOKEN")
    graph_version = os.environ.get("GRAPH_VERSION", "v25.0")
    graph_base = f"https://graph.facebook.com/{graph_version}"

    r2 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    try:
        existing = r2.get_object(Bucket=bucket, Key=R2_LEDGER_KEY)
        ledger = json.loads(existing["Body"].read().decode("utf-8"))
        write_receipt(ledger)
        print("Publicación ya registrada; no se repite.")
        return
    except r2.exceptions.NoSuchKey:
        pass
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {
            "NoSuchKey",
            "404",
        }:
            raise

    r2.put_object(
        Bucket=bucket,
        Key=R2_VIDEO_KEY,
        Body=VIDEO.read_bytes(),
        ContentType="video/mp4",
    )
    video_url = f"{public_base}/{R2_VIDEO_KEY}"
    public_check = requests.head(video_url, timeout=30, allow_redirects=True)
    public_check.raise_for_status()

    account = graph_get(
        graph_base,
        ig_user_id,
        {"fields": "id,username", "access_token": ig_token},
    )
    pending = {
        "schema_version": "supervised_instagram_publication_v1",
        "status": "container_requested",
        "platform": "instagram",
        "video_sha256": EXPECTED_SHA256,
        "source_url": "https://www.youtube.com/watch?v=G4sUx2nX5EQ",
        "account_id": account.get("id"),
        "account_username": account.get("username"),
        "requested_at": utc_now(),
    }
    r2.put_object(
        Bucket=bucket,
        Key=R2_LEDGER_KEY,
        Body=json.dumps(pending, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

    container = graph_post(
        graph_base,
        f"{ig_user_id}/media",
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": CAPTION,
            "share_to_feed": "true",
            "access_token": ig_token,
        },
    )
    creation_id = str(container["id"])
    pending["creation_id"] = creation_id
    pending["status"] = "processing"
    r2.put_object(
        Bucket=bucket,
        Key=R2_LEDGER_KEY,
        Body=json.dumps(pending, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        status_payload = graph_get(
            graph_base,
            creation_id,
            {"fields": "status_code", "access_token": ig_token},
        )
        status = str(status_payload.get("status_code", "")).upper()
        if status in {"FINISHED", "PUBLISHED"}:
            break
        if status in {"ERROR", "FAILED", "EXPIRED"}:
            raise RuntimeError(f"Instagram no procesó el video: {status_payload}")
        time.sleep(5)
    else:
        raise TimeoutError("Instagram no terminó de procesar el video")

    published = graph_post(
        graph_base,
        f"{ig_user_id}/media_publish",
        {"creation_id": creation_id, "access_token": ig_token},
    )
    media_id = str(published["id"])
    media = graph_get(
        graph_base,
        media_id,
        {
            "fields": "id,media_type,media_product_type,permalink,timestamp",
            "access_token": ig_token,
        },
    )
    receipt = {
        **pending,
        "status": "published",
        "media_id": media_id,
        "permalink": media.get("permalink"),
        "media_type": media.get("media_type"),
        "media_product_type": media.get("media_product_type"),
        "platform_timestamp": media.get("timestamp"),
        "verified_at": utc_now(),
    }
    r2.put_object(
        Bucket=bucket,
        Key=R2_LEDGER_KEY,
        Body=json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    write_receipt(receipt)
    print(f"Instagram Reel publicado: {receipt.get('permalink')}")


if __name__ == "__main__":
    main()
