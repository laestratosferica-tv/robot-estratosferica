#!/usr/bin/env python3
"""Publish one approved Reel through Meta APIs with fail-closed controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping



SCHEMA = "supervised_meta_publication_v1"
SUPPORTED_PLATFORMS = {"instagram", "facebook"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("manifest_schema_invalid")
    platform = str(payload.get("platform", "")).strip().lower()
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError("manifest_platform_must_be_single_meta_network")
    if not str(payload.get("approval_id", "")).strip():
        raise ValueError("manifest_approval_id_missing")
    if not str(payload.get("caption", "")).strip():
        raise ValueError("manifest_caption_missing")
    return payload


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[Path, str]:
    video = (repository_root / str(manifest.get("video_path", ""))).resolve()
    if repository_root.resolve() not in video.parents:
        raise ValueError("video_path_outside_repository")
    if not video.is_file():
        raise ValueError("approved_video_missing")
    actual_hash = sha256_file(video)
    if actual_hash != manifest.get("video_sha256"):
        raise ValueError("approved_video_hash_mismatch")
    return video, actual_hash


def required_environment(platform: str) -> tuple[str, ...]:
    common = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "R2_ENDPOINT_URL",
        "BUCKET_NAME",
        "R2_PUBLIC_BASE_URL",
    )
    if platform == "instagram":
        return common + ("IG_USER_ID", "IG_ACCESS_TOKEN")
    return common + ("FB_PAGE_ID", "FB_PAGE_ACCESS_TOKEN")


def validate_environment(
    platform: str,
    environment: Mapping[str, str],
) -> list[str]:
    return [name for name in required_environment(platform) if not environment.get(name, "").strip()]


def graph_post(base: str, path: str, *, data: dict[str, str]) -> dict[str, Any]:
    import requests

    response = requests.post(f"{base}/{path.lstrip('/')}", data=data, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError("meta_provider_rejected_operation")
    return payload


def graph_get(base: str, path: str, *, params: dict[str, str]) -> dict[str, Any]:
    import requests

    response = requests.get(f"{base}/{path.lstrip('/')}", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError("meta_provider_rejected_query")
    return payload


def graph_delete(base: str, path: str, *, data: dict[str, str]) -> dict[str, Any]:
    import requests

    response = requests.delete(f"{base}/{path.lstrip('/')}", data=data, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload or payload.get("success") is not True:
        raise RuntimeError("meta_provider_rejected_delete")
    return payload


def upload_public_video(video: Path, manifest: Mapping[str, Any], env: Mapping[str, str]) -> str:
    import boto3

    key = f"approved/meta/{manifest['platform']}/{manifest['slug']}-{manifest['video_sha256'][:16]}.mp4"
    client = boto3.client(
        "s3",
        endpoint_url=env["R2_ENDPOINT_URL"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    client.put_object(
        Bucket=env["BUCKET_NAME"],
        Key=key,
        Body=video.read_bytes(),
        ContentType="video/mp4",
    )
    return f"{env['R2_PUBLIC_BASE_URL'].rstrip('/')}/{key}"


def publish_instagram(video_url: str, caption: str, env: Mapping[str, str], graph_base: str) -> str:
    created = graph_post(
        graph_base,
        f"{env['IG_USER_ID']}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": env["IG_ACCESS_TOKEN"],
        },
    )
    creation_id = str(created["id"])
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        status_payload = graph_get(
            graph_base,
            creation_id,
            params={"fields": "status_code", "access_token": env["IG_ACCESS_TOKEN"]},
        )
        status = str(status_payload.get("status_code", "")).upper()
        if status in {"FINISHED", "PUBLISHED"}:
            break
        if status in {"ERROR", "FAILED", "EXPIRED"}:
            raise RuntimeError("instagram_container_failed")
        time.sleep(5)
    else:
        raise TimeoutError("instagram_container_timeout")
    published = graph_post(
        graph_base,
        f"{env['IG_USER_ID']}/media_publish",
        data={"creation_id": creation_id, "access_token": env["IG_ACCESS_TOKEN"]},
    )
    return str(published["id"])


def publish_facebook(video_url: str, caption: str, env: Mapping[str, str], graph_base: str) -> str:
    import requests

    start = graph_post(
        graph_base,
        f"{env['FB_PAGE_ID']}/video_reels",
        data={"upload_phase": "start", "access_token": env["FB_PAGE_ACCESS_TOKEN"]},
    )
    video_id = str(start["video_id"])
    upload = requests.post(
        str(start["upload_url"]),
        headers={
            "Authorization": f"OAuth {env['FB_PAGE_ACCESS_TOKEN']}",
            "file_url": video_url,
        },
        timeout=120,
    )
    upload.raise_for_status()
    graph_post(
        graph_base,
        f"{env['FB_PAGE_ID']}/video_reels",
        data={
            "upload_phase": "finish",
            "video_id": video_id,
            "video_state": "PUBLISHED",
            "description": caption[:2200],
            "access_token": env["FB_PAGE_ACCESS_TOKEN"],
        },
    )
    return video_id


def run(
    manifest_path: Path,
    *,
    repository_root: Path,
    environment: Mapping[str, str] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    manifest = load_manifest(manifest_path)
    video, video_hash = validate_manifest(manifest, repository_root=repository_root)
    platform = manifest["platform"]
    missing = validate_environment(platform, env)
    if missing:
        raise RuntimeError("missing_protected_configuration:" + ",".join(missing))

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "slug": manifest["slug"],
        "platform": platform,
        "video_sha256": video_hash,
        "approval_id": manifest["approval_id"],
        "dry_run": dry_run,
        "publishing_attempted": False,
        "published": False,
        "secret_values_exposed": False,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        return receipt

    if manifest.get("simulation_only") is True:
        raise RuntimeError("manifest_is_simulation_only")
    if env.get("PRODUCTION_ARMED") != "true":
        raise RuntimeError("production_not_armed")
    if env.get("PUBLICATION_APPROVAL_ID") != manifest["approval_id"]:
        raise RuntimeError("publication_approval_mismatch")

    video_url = upload_public_video(video, manifest, env)
    graph_version = env.get("GRAPH_VERSION", "v25.0")
    graph_base = f"https://graph.facebook.com/{graph_version}"
    receipt["publishing_attempted"] = True
    if platform == "instagram":
        media_id = publish_instagram(video_url, manifest["caption"], env, graph_base)
    else:
        media_id = publish_facebook(video_url, manifest["caption"], env, graph_base)
    receipt.update({"published": True, "media_id": media_id})

    # A Facebook replacement is deliberately post-publication only: the old
    # post remains untouched unless Graph confirms the replacement exists.
    replacement_post_id = str(manifest.get("replace_facebook_post_id", "")).strip()
    if replacement_post_id:
        if platform != "facebook":
            raise ValueError("facebook_replacement_requires_facebook_platform")
        confirmed = graph_get(
            graph_base,
            media_id,
            params={"fields": "id,permalink_url", "access_token": env["FB_PAGE_ACCESS_TOKEN"]},
        )
        if str(confirmed.get("id", "")) != media_id:
            raise RuntimeError("facebook_reel_verification_failed")
        receipt["permalink_url"] = str(confirmed.get("permalink_url", ""))
        graph_delete(
            graph_base,
            replacement_post_id,
            data={"access_token": env["FB_PAGE_ACCESS_TOKEN"]},
        )
        receipt.update(
            {
                "replaced_post_id": replacement_post_id,
                "replaced_post_deleted": True,
            }
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="artifacts/meta-supervised-publication.json")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    receipt = run(
        Path(args.manifest).resolve(),
        repository_root=root,
        dry_run=not args.live,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Validación segura completada." if not args.live else "Publicación supervisada completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
