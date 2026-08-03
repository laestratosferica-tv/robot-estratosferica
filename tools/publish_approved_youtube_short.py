#!/usr/bin/env python3
"""Publish one approved YouTube Short with an auditable, idempotent receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "approved_youtube_short_publication_v1"


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
    for key in ("slug", "approval_id", "title", "description", "video_path", "video_sha256"):
        if not str(manifest.get(key, "")).strip():
            raise ValueError(f"manifest_{key}_missing")
    if len(str(manifest["title"])) > 95 or len(str(manifest["description"])) > 4500:
        raise ValueError("manifest_copy_length_invalid")
    if manifest.get("privacy_status") != "public":
        raise ValueError("manifest_must_be_explicitly_public")
    video = (root / str(manifest["video_path"])).resolve()
    if root.resolve() not in video.parents or video.suffix.lower() != ".mp4" or not video.is_file():
        raise ValueError("approved_video_missing")
    if digest(video) != manifest["video_sha256"]:
        raise ValueError("approved_video_hash_mismatch")
    return manifest


def storage(env: Mapping[str, str]):
    import boto3

    required(env, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL", "BUCKET_NAME")
    return boto3.client(
        "s3",
        endpoint_url=env["R2_ENDPOINT_URL"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def read_ledger(client: Any, env: Mapping[str, str], key: str) -> dict[str, Any] | None:
    try:
        return json.loads(client.get_object(Bucket=env["BUCKET_NAME"], Key=key)["Body"].read().decode("utf-8"))
    except client.exceptions.NoSuchKey:
        return None
    except Exception as error:
        response = getattr(error, "response", {})
        if response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return None
        raise


def write_ledger(client: Any, env: Mapping[str, str], key: str, payload: Mapping[str, Any]) -> None:
    client.put_object(
        Bucket=env["BUCKET_NAME"],
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def publish(video: Path, manifest: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    credentials = Credentials(
        None,
        refresh_token=env["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=env["YOUTUBE_CLIENT_ID"],
        client_secret=env["YOUTUBE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": manifest["title"],
                "description": manifest["description"],
                "categoryId": "20",
            },
            "status": {
                "privacyStatus": manifest["privacy_status"],
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(str(video), chunksize=-1, resumable=True, mimetype="video/mp4"),
    )
    response: dict[str, Any] | None = None
    while response is None:
        _, response = request.next_chunk()
    video_id = str(response["id"])
    verified = youtube.videos().list(part="id,status", id=video_id).execute()
    items = verified.get("items", [])
    if not items or str(items[0].get("id")) != video_id or items[0].get("status", {}).get("privacyStatus") != "public":
        raise RuntimeError("youtube_publication_verification_failed")
    return {"video_id": video_id, "permalink": f"https://www.youtube.com/shorts/{video_id}", "privacy_status": "public"}


def run(manifest_path: Path, *, root: Path, live: bool, env: Mapping[str, str]) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, root)
    video = (root / manifest["video_path"]).resolve()
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "slug": manifest["slug"],
        "approval_id": manifest["approval_id"],
        "video_sha256": manifest["video_sha256"],
        "copy_verified": {"title": manifest["title"], "description": manifest["description"]},
        "live": live,
        "publishing_attempted": False,
        "published": False,
        "secret_values_exposed": False,
        "checked_at": now(),
    }
    if not live:
        return receipt
    if env.get("PRODUCTION_ARMED") != "true" or env.get("PUBLICATION_APPROVAL_ID") != manifest["approval_id"]:
        raise RuntimeError("production_approval_not_armed")
    required(env, "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")
    client = storage(env)
    key = f"publication-ledger/youtube-short/{manifest['slug']}.json"
    existing = read_ledger(client, env, key)
    if existing and existing.get("status") == "published":
        return {**receipt, "published": True, "already_recorded": True, "publication": existing}
    pending = {"schema": SCHEMA, "status": "claim_created", "slug": manifest["slug"], "approval_id": manifest["approval_id"], "video_sha256": manifest["video_sha256"], "requested_at": now()}
    write_ledger(client, env, key, pending)
    receipt["publishing_attempted"] = True
    try:
        result = publish(video, manifest, env)
    except Exception as error:
        failed = {**pending, "status": "blocked_or_failed", "error_type": type(error).__name__, "error": str(error)[:180], "failed_at": now()}
        write_ledger(client, env, key, failed)
        raise
    final = {**pending, "status": "published", **result, "verified_at": now()}
    write_ledger(client, env, key, final)
    return {**receipt, "published": True, "publication": final}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="artifacts/youtube-short-publication-result.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    receipt = run((root / args.manifest).resolve(), root=root, live=args.live, env=os.environ)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("YouTube Short publicado." if args.live else "Compuerta de solo lectura completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
