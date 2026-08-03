#!/usr/bin/env python3
"""Run due, pre-approved Meta publications with fail-closed idempotency."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from tools.publish_supervised_meta import load_manifest, run as publish_manifest
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from publish_supervised_meta import load_manifest, run as publish_manifest


QUEUE_SCHEMA = "scheduled_publication_queue_v1"
STATE_SCHEMA = "scheduled_publication_state_v1"
DEFAULT_STATE_KEY = "publishing/state/scheduled-publications-v1.json"


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("publish_at_timezone_required")
    return parsed.astimezone(timezone.utc)


def load_queue(path: Path) -> dict[str, Any]:
    queue = json.loads(path.read_text(encoding="utf-8"))
    if queue.get("schema") != QUEUE_SCHEMA:
        raise ValueError("queue_schema_invalid")
    items = queue.get("items")
    if not isinstance(items, list):
        raise ValueError("queue_items_invalid")
    seen: set[str] = set()
    for item in items:
        content_id = str(item.get("content_id", "")).strip()
        if not content_id or content_id in seen:
            raise ValueError("queue_content_id_missing_or_duplicate")
        seen.add(content_id)
        parse_time(str(item.get("publish_at", "")))
        if item.get("status") != "approved":
            raise ValueError("queue_item_not_approved")
        if not str(item.get("manifest_path", "")).strip():
            raise ValueError("queue_manifest_path_missing")
        if not str(item.get("approval_id", "")).strip():
            raise ValueError("queue_approval_id_missing")
    return queue


def empty_state() -> dict[str, Any]:
    return {"schema": STATE_SCHEMA, "items": {}}


def r2_client(env: Mapping[str, str]):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=env["R2_ENDPOINT_URL"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def load_remote_state(env: Mapping[str, str], key: str) -> dict[str, Any]:
    client = r2_client(env)
    try:
        response = client.get_object(Bucket=env["BUCKET_NAME"], Key=key)
    except client.exceptions.NoSuchKey:
        return empty_state()
    except Exception as exc:
        response = getattr(exc, "response", {})
        if response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return empty_state()
        raise
    state = json.loads(response["Body"].read().decode("utf-8"))
    if state.get("schema") != STATE_SCHEMA or not isinstance(state.get("items"), dict):
        raise ValueError("remote_state_invalid")
    return state


def save_remote_state(env: Mapping[str, str], key: str, state: Mapping[str, Any]) -> None:
    r2_client(env).put_object(
        Bucket=env["BUCKET_NAME"],
        Key=key,
        Body=(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def due_items(
    queue: Mapping[str, Any], state: Mapping[str, Any], *, now: datetime
) -> list[dict[str, Any]]:
    completed = state.get("items", {})
    due = [
        item
        for item in queue["items"]
        if item.get("enabled", True)
        and parse_time(item["publish_at"]) <= now
        and completed.get(item["content_id"], {}).get("status")
        not in {"publishing", "published"}
    ]
    return sorted(due, key=lambda item: (parse_time(item["publish_at"]), item["content_id"]))


def validate_item(item: Mapping[str, Any], repository_root: Path) -> Path:
    manifest_path = (repository_root / item["manifest_path"]).resolve()
    if repository_root.resolve() not in manifest_path.parents:
        raise ValueError("manifest_path_outside_repository")
    manifest = load_manifest(manifest_path)
    if manifest["approval_id"] != item["approval_id"]:
        raise ValueError("queue_manifest_approval_mismatch")
    return manifest_path


def execute_queue(
    queue_path: Path,
    *,
    repository_root: Path,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
    live: bool = False,
    max_items: int = 3,
) -> dict[str, Any]:
    env = dict(os.environ if environment is None else environment)
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    queue = load_queue(queue_path)
    state_key = env.get("SCHEDULED_PUBLICATION_STATE_KEY", DEFAULT_STATE_KEY)

    if live:
        if env.get("PRODUCTION_ARMED") != "true":
            raise RuntimeError("production_not_armed")
        if env.get("SCHEDULED_PUBLISHING_ARMED") != "true":
            raise RuntimeError("scheduled_publishing_not_armed")
        state = load_remote_state(env, state_key)
    else:
        state = empty_state()

    selected = due_items(queue, state, now=current_time)[:max_items]
    report: dict[str, Any] = {
        "schema": QUEUE_SCHEMA,
        "live": live,
        "checked_at": current_time.isoformat(),
        "due_count": len(selected),
        "results": [],
        "secret_values_exposed": False,
    }
    for item in selected:
        manifest_path = validate_item(item, repository_root)
        if live:
            state["items"][item["content_id"]] = {
                "status": "publishing",
                "started_at": current_time.isoformat(),
                "approval_id": item["approval_id"],
            }
            save_remote_state(env, state_key, state)

        item_env = {**env, "PUBLICATION_APPROVAL_ID": item["approval_id"]}
        try:
            receipt = publish_manifest(
                manifest_path,
                repository_root=repository_root,
                environment=item_env,
                dry_run=not live,
            )
        except Exception:
            if live:
                state["items"][item["content_id"]]["status"] = "failed"
                state["items"][item["content_id"]]["failed_at"] = datetime.now(timezone.utc).isoformat()
                save_remote_state(env, state_key, state)
            raise

        report["results"].append(
            {
                "content_id": item["content_id"],
                "platform": receipt["platform"],
                "published": receipt["published"],
                "dry_run": receipt["dry_run"],
                "media_id": receipt.get("media_id"),
            }
        )
        if live:
            state["items"][item["content_id"]] = {
                "status": "published",
                "published_at": datetime.now(timezone.utc).isoformat(),
                "approval_id": item["approval_id"],
                "media_id": receipt.get("media_id"),
            }
            save_remote_state(env, state_key, state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="config/scheduled_publications_v1.json")
    parser.add_argument("--output", default="artifacts/scheduled-publication-run.json")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-items", type=int, default=3)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    report = execute_queue(
        Path(args.queue).resolve(),
        repository_root=root,
        live=args.live,
        max_items=args.max_items,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Cola revisada: {report['due_count']} pieza(s) vencida(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
