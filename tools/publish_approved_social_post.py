#!/usr/bin/env python3
"""Publish an approved image, text/link post, or non-interactive Story."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "approved_social_post_v1"
SUPPORTED = {
    ("instagram", "image"),
    ("instagram", "story_image"),
    ("facebook", "image"),
    ("facebook", "text_link"),
    ("threads", "image"),
    ("threads", "video"),
    ("threads", "text"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def required(env: Mapping[str, str], *names: str) -> None:
    missing = [name for name in names if not env.get(name, "").strip()]
    if missing:
        raise RuntimeError("missing_protected_configuration:" + ",".join(missing))


def load_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("manifest_schema_invalid")
    platform = str(manifest.get("platform", "")).lower()
    post_type = str(manifest.get("post_type", "")).lower()
    if (platform, post_type) not in SUPPORTED:
        raise ValueError("manifest_platform_format_unsupported")
    for field in ("slug", "approval_id", "text"):
        if not str(manifest.get(field, "")).strip():
            raise ValueError(f"manifest_{field}_missing")
    if manifest.get("interactive_sticker_required") is True:
        raise ValueError("interactive_story_requires_native_manual_step")
    needs_asset = post_type in {"image", "story_image", "video"}
    if needs_asset:
        asset = (root / str(manifest.get("asset_path", ""))).resolve()
        if root.resolve() not in asset.parents or not asset.is_file():
            raise ValueError("approved_asset_missing")
        if digest(asset) != manifest.get("asset_sha256"):
            raise ValueError("approved_asset_hash_mismatch")
    if post_type == "text_link":
        link = str(manifest.get("link", ""))
        if not link.startswith("https://"):
            raise ValueError("approved_https_link_required")
    return manifest


def graph_request(method: str, base: str, path: str, *, token: str, data: dict[str, str] | None = None, params: dict[str, str] | None = None) -> dict[str, Any]:
    import requests

    response = requests.request(
        method,
        f"{base.rstrip('/')}/{path.lstrip('/')}",
        headers={"Authorization": f"Bearer {token}"},
        data=data,
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError("provider_rejected_operation")
    return payload


def storage(env: Mapping[str, str]):
    import boto3

    required(env, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL", "BUCKET_NAME", "R2_PUBLIC_BASE_URL")
    return boto3.client(
        "s3",
        endpoint_url=env["R2_ENDPOINT_URL"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_asset(manifest: Mapping[str, Any], root: Path, env: Mapping[str, str]) -> str:
    path = root / manifest["asset_path"]
    extension = path.suffix.lower()
    content_type = "video/mp4" if extension == ".mp4" else ("image/png" if extension == ".png" else "image/jpeg")
    key = f"approved/social/{manifest['slug']}-{manifest['asset_sha256'][:16]}{extension}"
    storage(env).put_object(Bucket=env["BUCKET_NAME"], Key=key, Body=path.read_bytes(), ContentType=content_type)
    return f"{env['R2_PUBLIC_BASE_URL'].rstrip('/')}/{key}"


def wait_instagram(base: str, container: str, token: str) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        payload = graph_request("GET", base, container, token=token, params={"fields": "status_code"})
        status = str(payload.get("status_code", "")).upper()
        if status in {"FINISHED", "PUBLISHED"}:
            return
        if status in {"ERROR", "FAILED", "EXPIRED"}:
            raise RuntimeError("instagram_container_failed")
        time.sleep(5)
    raise TimeoutError("instagram_container_timeout")


def publish_instagram(manifest: Mapping[str, Any], url: str, env: Mapping[str, str]) -> dict[str, Any]:
    required(env, "IG_USER_ID", "IG_ACCESS_TOKEN")
    base = f"https://graph.facebook.com/{env.get('GRAPH_VERSION', 'v25.0')}"
    data = {"image_url": url, "caption": manifest["text"]}
    if manifest["post_type"] == "story_image":
        data = {"image_url": url, "media_type": "STORIES"}
    created = graph_request("POST", base, f"{env['IG_USER_ID']}/media", token=env["IG_ACCESS_TOKEN"], data=data)
    wait_instagram(base, str(created["id"]), env["IG_ACCESS_TOKEN"])
    published = graph_request("POST", base, f"{env['IG_USER_ID']}/media_publish", token=env["IG_ACCESS_TOKEN"], data={"creation_id": str(created["id"])})
    return {"media_id": str(published["id"])}


def publish_facebook(manifest: Mapping[str, Any], url: str | None, env: Mapping[str, str]) -> dict[str, Any]:
    required(env, "FB_PAGE_ID", "FB_PAGE_ACCESS_TOKEN")
    base = f"https://graph.facebook.com/{env.get('GRAPH_VERSION', 'v25.0')}"
    if manifest["post_type"] == "image":
        result = graph_request("POST", base, f"{env['FB_PAGE_ID']}/photos", token=env["FB_PAGE_ACCESS_TOKEN"], data={"url": str(url), "caption": manifest["text"]})
    else:
        result = graph_request("POST", base, f"{env['FB_PAGE_ID']}/feed", token=env["FB_PAGE_ACCESS_TOKEN"], data={"message": manifest["text"], "link": manifest["link"]})
    return {"media_id": str(result.get("post_id") or result["id"])}


def publish_threads(manifest: Mapping[str, Any], url: str | None, env: Mapping[str, str]) -> dict[str, Any]:
    required(env, "THREADS_USER_ID", "THREADS_USER_ACCESS_TOKEN")
    base = f"https://graph.threads.net/{env.get('THREADS_GRAPH_VERSION', 'v1.0')}"
    data = {"media_type": "TEXT", "text": manifest["text"]}
    if manifest["post_type"] == "image":
        data = {"media_type": "IMAGE", "image_url": str(url), "text": manifest["text"]}
    elif manifest["post_type"] == "video":
        data = {"media_type": "VIDEO", "video_url": str(url), "text": manifest["text"]}
    created = graph_request("POST", base, f"{env['THREADS_USER_ID']}/threads", token=env["THREADS_USER_ACCESS_TOKEN"], data=data)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        status = graph_request(
            "GET",
            base,
            str(created["id"]),
            token=env["THREADS_USER_ACCESS_TOKEN"],
            params={"fields": "status,error_message"},
        )
        state = str(status.get("status", "")).upper()
        if state in {"FINISHED", "PUBLISHED"}:
            break
        if state in {"ERROR", "FAILED", "EXPIRED"}:
            raise RuntimeError("threads_container_failed")
        time.sleep(2)
    else:
        raise TimeoutError("threads_container_timeout")
    published = graph_request("POST", base, f"{env['THREADS_USER_ID']}/threads_publish", token=env["THREADS_USER_ACCESS_TOKEN"], data={"creation_id": str(created["id"])})
    return {"media_id": str(published["id"])}


def run(manifest_path: Path, *, root: Path, live: bool, env: Mapping[str, str]) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, root)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "slug": manifest["slug"],
        "platform": manifest["platform"],
        "post_type": manifest["post_type"],
        "approval_id": manifest["approval_id"],
        "live": live,
        "published": False,
        "publishing_attempted": False,
        "secret_values_exposed": False,
        "checked_at": now(),
    }
    if not live:
        return receipt
    if env.get("PRODUCTION_ARMED") != "true" or env.get("PUBLICATION_APPROVAL_ID") != manifest["approval_id"]:
        raise RuntimeError("production_approval_not_armed")
    url = upload_asset(manifest, root, env) if manifest["post_type"] in {"image", "story_image", "video"} else None
    receipt["publishing_attempted"] = True
    if manifest["platform"] == "instagram":
        result = publish_instagram(manifest, str(url), env)
    elif manifest["platform"] == "facebook":
        result = publish_facebook(manifest, url, env)
    else:
        result = publish_threads(manifest, url, env)
    receipt.update({"published": True, **result})
    return receipt
