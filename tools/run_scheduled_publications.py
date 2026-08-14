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

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tools.publish_approved_carousel import run as publish_carousel
    from tools.publish_approved_social_post import run as publish_social_post
    from tools.publish_approved_youtube_short import run as publish_youtube
    from tools.publish_supervised_meta import run as publish_meta_reel
except ModuleNotFoundError:
    from publish_approved_carousel import run as publish_carousel
    from publish_approved_social_post import run as publish_social_post
    from publish_approved_youtube_short import run as publish_youtube
    from publish_supervised_meta import run as publish_meta_reel


QUEUE_SCHEMA = "scheduled_publication_queue_v1"
STATE_SCHEMA = "scheduled_publication_state_v1"
DEFAULT_STATE_KEY = "publishing/state/scheduled-publications-v1.json"
COMMERCIAL_CHECKS = {
    "asset_final",
}


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
    fun_sources: dict[str, set[str]] = {}
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
        if item.get("commercial") is True:
            checks = item.get("commercial_checks", {})
            if not isinstance(checks, dict) or any(
                checks.get(name) is not True for name in COMMERCIAL_CHECKS
            ):
                raise ValueError("commercial_publication_checks_incomplete")
            if not str(item.get("resolver_evidence_path", "")).strip():
                raise ValueError("commercial_resolver_evidence_missing")
        if item.get("category") == "contenido_divertido" and item.get("enabled", True):
            repository_root = path.parent.parent if path.parent.name == "config" else path.parent
            evidence_path = repository_root / str(item.get("license_evidence_path", ""))
            if not evidence_path.is_file():
                raise ValueError("fun_content_license_evidence_missing")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            source = str(evidence.get("source_page", "")).strip()
            if not source:
                raise ValueError("fun_content_source_page_missing")
            publish_day = parse_time(str(item["publish_at"])).date().isoformat()
            fun_sources.setdefault(source, set()).add(publish_day)
            item["_fun_source_page"] = source

    repeated_sources = {
        source for source, publish_days in fun_sources.items() if len(publish_days) > 1
    }
    for item in items:
        if item.get("_fun_source_page") in repeated_sources:
            item["_blocked_by_repeat_guard"] = True
    queue["repeat_guard"] = {
        "blocked_sources": sorted(repeated_sources),
        "blocked_items": sum(
            1 for item in items if item.get("_blocked_by_repeat_guard") is True
        ),
    }
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
        and item.get("_blocked_by_repeat_guard") is not True
        and parse_time(item["publish_at"]) <= now
        and completed.get(item["content_id"], {}).get("status")
        not in {"publishing", "published"}
    ]
    return sorted(due, key=lambda item: (parse_time(item["publish_at"]), item["content_id"]))


def validate_item(item: Mapping[str, Any], repository_root: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = (repository_root / item["manifest_path"]).resolve()
    if repository_root.resolve() not in manifest_path.parents:
        raise ValueError("manifest_path_outside_repository")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["approval_id"] != item["approval_id"]:
        raise ValueError("queue_manifest_approval_mismatch")
    if manifest.get("schema") not in {
        "supervised_meta_publication_v1",
        "approved_carousel_publication_v1",
        "approved_youtube_short_publication_v1",
        "approved_social_post_v1",
    }:
        raise ValueError("queue_manifest_schema_unsupported")
    if item.get("commercial") is True:
        evidence_path = (repository_root / item["resolver_evidence_path"]).resolve()
        if repository_root.resolve() not in evidence_path.parents:
            raise ValueError("resolver_evidence_path_outside_repository")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        required = {
            "publishable",
            "associate_tag_verified",
            "https_verified",
            "availability_verified",
            "product_match_verified",
            "visual_reference_verified",
            "approval_verified",
        }
        allowed_sources = {"amazon_paapi5", "amazon_listing_plus_editorial_approval"}
        if evidence.get("schema") != "amazon_affiliate_resolution_v1" or evidence.get(
            "source"
        ) not in allowed_sources or any(
            evidence.get(name) is not True for name in required
        ) or not str(evidence.get("affiliate_disclosure", "")).strip():
            raise ValueError("commercial_resolver_evidence_not_publishable")
        approval_path = (repository_root / str(evidence.get("approval_record_path", ""))).resolve()
        if evidence.get("source") == "amazon_listing_plus_editorial_approval" and (
            repository_root.resolve() not in approval_path.parents or not approval_path.is_file()
        ):
            raise ValueError("commercial_approval_record_missing")
    return manifest_path, manifest


def publish_item(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    environment: Mapping[str, str],
    live: bool,
) -> dict[str, Any]:
    schema = manifest["schema"]
    if schema == "supervised_meta_publication_v1":
        return publish_meta_reel(
            manifest_path,
            repository_root=repository_root,
            environment=environment,
            dry_run=not live,
        )
    if schema == "approved_carousel_publication_v1":
        receipt = publish_carousel(
            manifest_path, root=repository_root, live=live, env=environment
        )
        if live and any(
            result.get("status") not in {"published", "already_recorded"}
            for result in receipt.get("platforms", {}).values()
        ):
            raise RuntimeError("carousel_platform_publication_incomplete")
        receipt["published"] = live
        return receipt
    if schema == "approved_youtube_short_publication_v1":
        return publish_youtube(
            manifest_path, root=repository_root, live=live, env=environment
        )
    return publish_social_post(
        manifest_path, root=repository_root, live=live, env=environment
    )


def execute_queue(
    queue_path: Path,
    *,
    repository_root: Path,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
    live: bool = False,
    max_items: int = 10,
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
        "failed_count": 0,
        "repeat_guard": queue.get("repeat_guard", {}),
        "secret_values_exposed": False,
    }
    for item in selected:
        manifest_path, manifest = validate_item(item, repository_root)
        if live:
            state["items"][item["content_id"]] = {
                "status": "publishing",
                "started_at": current_time.isoformat(),
                "approval_id": item["approval_id"],
            }
            save_remote_state(env, state_key, state)

        item_env = {**env, "PUBLICATION_APPROVAL_ID": item["approval_id"]}
        try:
            receipt = publish_item(
                manifest_path,
                manifest,
                repository_root=repository_root,
                environment=item_env,
                live=live,
            )
        except Exception as error:
            if live:
                state["items"][item["content_id"]]["status"] = "failed"
                state["items"][item["content_id"]]["failed_at"] = datetime.now(timezone.utc).isoformat()
                state["items"][item["content_id"]]["error_type"] = type(error).__name__
                save_remote_state(env, state_key, state)
            report["failed_count"] += 1
            report["results"].append(
                {
                    "content_id": item["content_id"],
                    "schema": manifest["schema"],
                    "platform": manifest.get("platform", "multi"),
                    "published": False,
                    "dry_run": not live,
                    "error_type": type(error).__name__,
                    "error": str(error)[:240],
                }
            )
            continue

        report["results"].append(
            {
                "content_id": item["content_id"],
                "schema": manifest["schema"],
                "platform": receipt.get("platform", manifest.get("platform", "multi")),
                "published": bool(receipt.get("published")),
                "dry_run": not live,
                "media_id": receipt.get("media_id"),
                "platforms": receipt.get("platforms"),
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
    parser.add_argument("--max-items", type=int, default=10)
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
    print(
        f"Cola revisada: {report['due_count']} pieza(s) vencida(s), "
        f"{report['failed_count']} fallo(s)."
    )
    return 1 if report["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
