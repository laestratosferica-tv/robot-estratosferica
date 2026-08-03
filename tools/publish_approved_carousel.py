#!/usr/bin/env python3
"""Publish one explicitly approved seven-image carousel, fail-closed per platform."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import boto3
import requests
from botocore.exceptions import ClientError


SCHEMA = "approved_carousel_publication_v1"
PLATFORMS = ("instagram", "facebook", "threads")
MAX_IMAGES = 10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not_png:{path}")
    return struct.unpack(">II", header[16:24])


def required(env: Mapping[str, str], *names: str) -> None:
    missing = [name for name in names if not env.get(name, "").strip()]
    if missing:
        raise RuntimeError("missing_protected_configuration:" + ",".join(missing))


def request(method: str, base: str, path: str, *, token: str, data: dict[str, str] | None = None, params: dict[str, str] | None = None) -> dict[str, Any]:
    response = requests.request(
        method, f"{base.rstrip('/')}/{path.lstrip('/')}", data=data, params=params,
        headers={"Authorization": f"Bearer {token}"}, timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError("provider_rejected_operation")
    return payload


def wait_ready(base: str, container: str, token: str, *, threads: bool = False) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        fields = "status,error_message" if threads else "status_code"
        payload = request("GET", base, container, token=token, params={"fields": fields})
        status = str(payload.get("status" if threads else "status_code", "")).upper()
        if status in {"FINISHED", "PUBLISHED"}:
            return
        if status in {"ERROR", "FAILED", "EXPIRED"}:
            raise RuntimeError("container_failed")
        time.sleep(5)
    raise TimeoutError("container_timeout")


def load_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("platforms") != list(PLATFORMS):
        raise ValueError("manifest_scope_invalid")
    if not str(manifest.get("approval_id", "")).strip() or not str(manifest.get("caption", "")).strip() or not str(manifest.get("threads_text", "")).strip():
        raise ValueError("manifest_approval_or_copy_missing")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != 7:
        raise ValueError("manifest_must_contain_exactly_7_assets")
    for index, asset in enumerate(assets, start=1):
        if asset.get("order") != index:
            raise ValueError("manifest_asset_order_invalid")
        file_path = (root / str(asset.get("path", ""))).resolve()
        if root.resolve() not in file_path.parents or not file_path.is_file():
            raise ValueError("approved_asset_missing")
        if digest(file_path) != asset.get("sha256"):
            raise ValueError("approved_asset_hash_mismatch")
        if png_size(file_path) != (1080, 1350):
            raise ValueError("approved_asset_dimensions_invalid")
    return manifest


def account_preflight(manifest: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    required(env, "IG_USER_ID", "IG_ACCESS_TOKEN", "FB_PAGE_ID", "FB_PAGE_ACCESS_TOKEN", "THREADS_USER_ID", "THREADS_USER_ACCESS_TOKEN")
    graph = f"https://graph.facebook.com/{env.get('GRAPH_VERSION', 'v25.0')}"
    threads = f"https://graph.threads.net/{env.get('THREADS_GRAPH_VERSION', 'v1.0')}"
    expected = manifest["expected_accounts"]
    checks = {
        "instagram": (graph, env["IG_USER_ID"], env["IG_ACCESS_TOKEN"], {"fields": "id,username"}),
        "facebook": (graph, env["FB_PAGE_ID"], env["FB_PAGE_ACCESS_TOKEN"], {"fields": "id,name,link"}),
        "threads": (threads, env["THREADS_USER_ID"], env["THREADS_USER_ACCESS_TOKEN"], {"fields": "id,username"}),
    }
    accounts: dict[str, Any] = {}
    for platform, (base, identifier, token, fields) in checks.items():
        try:
            profile = request("GET", base, identifier, token=token, params=fields)
            actual = profile.get("name") if platform == "facebook" else profile.get("username")
            wanted = expected["facebook_page_name"] if platform == "facebook" else expected[f"{platform}_username"]
            if str(actual or "").strip().lower() != str(wanted).strip().lower():
                raise RuntimeError(f"{platform}_account_mismatch")
            accounts[platform] = {"status": "official_account_confirmed", "id": profile.get("id"), "username": profile.get("username"), "name": profile.get("name"), "link": profile.get("link")}
        except Exception as error:
            accounts[platform] = {"status": "blocked", "error_type": type(error).__name__, "error": str(error)[:180]}
    return accounts


def storage(env: Mapping[str, str]):
    required(env, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL", "BUCKET_NAME", "R2_PUBLIC_BASE_URL")
    return boto3.client("s3", endpoint_url=env["R2_ENDPOINT_URL"], aws_access_key_id=env["AWS_ACCESS_KEY_ID"], aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"], region_name="auto")


def read_ledger(client: Any, env: Mapping[str, str], key: str) -> dict[str, Any] | None:
    try:
        return json.loads(client.get_object(Bucket=env["BUCKET_NAME"], Key=key)["Body"].read().decode("utf-8"))
    except client.exceptions.NoSuchKey:
        return None
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return None
        raise


def write_ledger(client: Any, env: Mapping[str, str], key: str, payload: Mapping[str, Any]) -> None:
    client.put_object(Bucket=env["BUCKET_NAME"], Key=key, Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"), ContentType="application/json")


def upload_images(client: Any, env: Mapping[str, str], manifest: Mapping[str, Any], root: Path) -> list[str]:
    urls: list[str] = []
    for asset in manifest["assets"]:
        path = root / asset["path"]
        key = f"approved/carousels/{manifest['slug']}/{asset['order']}-{asset['sha256'][:16]}.png"
        client.put_object(Bucket=env["BUCKET_NAME"], Key=key, Body=path.read_bytes(), ContentType="image/png")
        url = f"{env['R2_PUBLIC_BASE_URL'].rstrip('/')}/{key}"
        check = requests.head(url, allow_redirects=True, timeout=30)
        check.raise_for_status()
        urls.append(url)
    return urls


def publish_instagram(urls: list[str], caption: str, env: Mapping[str, str]) -> dict[str, Any]:
    base = f"https://graph.facebook.com/{env.get('GRAPH_VERSION', 'v25.0')}"; token = env["IG_ACCESS_TOKEN"]
    children = []
    for url in urls:
        child = request("POST", base, f"{env['IG_USER_ID']}/media", token=token, data={"image_url": url, "is_carousel_item": "true"})
        wait_ready(base, str(child["id"]), token); children.append(str(child["id"]))
    parent = request("POST", base, f"{env['IG_USER_ID']}/media", token=token, data={"media_type": "CAROUSEL", "children": ",".join(children), "caption": caption})
    wait_ready(base, str(parent["id"]), token)
    published = request("POST", base, f"{env['IG_USER_ID']}/media_publish", token=token, data={"creation_id": str(parent["id"])})
    media = request("GET", base, str(published["id"]), token=token, params={"fields": "id,permalink,media_type,media_product_type,timestamp"})
    return {"media_id": media.get("id"), "permalink": media.get("permalink"), "media_type": media.get("media_type"), "media_product_type": media.get("media_product_type"), "timestamp": media.get("timestamp")}


def publish_facebook(urls: list[str], caption: str, env: Mapping[str, str]) -> dict[str, Any]:
    base = f"https://graph.facebook.com/{env.get('GRAPH_VERSION', 'v25.0')}"; token = env["FB_PAGE_ACCESS_TOKEN"]
    attached: dict[str, str] = {}
    for index, url in enumerate(urls):
        photo = request("POST", base, f"{env['FB_PAGE_ID']}/photos", token=token, data={"url": url, "published": "false"})
        attached[f"attached_media[{index}]"] = json.dumps({"media_fbid": str(photo["id"])})
    post = request("POST", base, f"{env['FB_PAGE_ID']}/feed", token=token, data={**attached, "message": caption})
    media = request("GET", base, str(post["id"]), token=token, params={"fields": "id,permalink_url,message,created_time"})
    return {"media_id": media.get("id"), "permalink": media.get("permalink_url"), "timestamp": media.get("created_time")}


def publish_threads(urls: list[str], text: str, env: Mapping[str, str]) -> dict[str, Any]:
    base = f"https://graph.threads.net/{env.get('THREADS_GRAPH_VERSION', 'v1.0')}"; token = env["THREADS_USER_ACCESS_TOKEN"]
    children = []
    for url in urls:
        child = request("POST", base, f"{env['THREADS_USER_ID']}/threads", token=token, data={"media_type": "IMAGE", "image_url": url, "is_carousel_item": "true"})
        wait_ready(base, str(child["id"]), token, threads=True); children.append(str(child["id"]))
    parent = request("POST", base, f"{env['THREADS_USER_ID']}/threads", token=token, data={"media_type": "CAROUSEL", "children": ",".join(children), "text": text})
    wait_ready(base, str(parent["id"]), token, threads=True)
    published = request("POST", base, f"{env['THREADS_USER_ID']}/threads_publish", token=token, data={"creation_id": str(parent["id"])})
    media = request("GET", base, str(published["id"]), token=token, params={"fields": "id,permalink,media_type,text,timestamp"})
    return {"media_id": media.get("id"), "permalink": media.get("permalink"), "media_type": media.get("media_type"), "timestamp": media.get("timestamp")}


def run(manifest_path: Path, *, root: Path, live: bool, env: Mapping[str, str]) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, root)
    receipt: dict[str, Any] = {"schema": SCHEMA, "slug": manifest["slug"], "approval_id": manifest["approval_id"], "image_count": 7, "image_size": "1080x1350", "copy_verified": {"caption": manifest["caption"], "threads_text": manifest["threads_text"]}, "live": live, "publishing_attempted": False, "accounts": account_preflight(manifest, env), "platforms": {}, "checked_at": now(), "secret_values_exposed": False}
    if not live:
        return receipt
    if env.get("PRODUCTION_ARMED") != "true" or env.get("PUBLICATION_APPROVAL_ID") != manifest["approval_id"]:
        raise RuntimeError("production_approval_not_armed")
    client = storage(env); urls = upload_images(client, env, manifest, root)
    publishers = {"instagram": lambda: publish_instagram(urls, manifest["caption"], env), "facebook": lambda: publish_facebook(urls, manifest["caption"], env), "threads": lambda: publish_threads(urls, manifest["threads_text"], env)}
    for platform in PLATFORMS:
        key = f"publication-ledger/carousel/{manifest['slug']}/{platform}.json"; existing = read_ledger(client, env, key)
        if receipt["accounts"][platform]["status"] != "official_account_confirmed":
            receipt["platforms"][platform] = {"status": "blocked_preflight", "reason": receipt["accounts"][platform]}; continue
        if existing:
            receipt["platforms"][platform] = {"status": "already_recorded", "record": existing}; continue
        pending = {"schema": SCHEMA, "status": "claim_created", "slug": manifest["slug"], "platform": platform, "approval_id": manifest["approval_id"], "image_count": 7, "asset_hashes": [item["sha256"] for item in manifest["assets"]], "requested_at": now()}
        write_ledger(client, env, key, pending); receipt["publishing_attempted"] = True
        try:
            result = publishers[platform](); final = {**pending, "status": "published", **result, "verified_at": now()}; write_ledger(client, env, key, final); receipt["platforms"][platform] = final
        except Exception as error:
            failed = {**pending, "status": "blocked_or_failed", "error_type": type(error).__name__, "error": str(error)[:180], "failed_at": now()}; write_ledger(client, env, key, failed); receipt["platforms"][platform] = failed
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True); parser.add_argument("--live", action="store_true"); parser.add_argument("--output", default="artifacts/carousel-publication-result.json"); args = parser.parse_args()
    root = Path.cwd().resolve(); receipt = run((root / args.manifest).resolve(), root=root, live=args.live, env=os.environ)
    output = root / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Publicación completada." if args.live else "Compuerta de solo lectura completada."); return 0


if __name__ == "__main__":
    raise SystemExit(main())
