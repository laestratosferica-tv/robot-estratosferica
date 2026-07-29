#!/usr/bin/env python3
"""Collect one Instagram Reel's private insights without mutating Instagram."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from media_factory.metrics import build_reel_learning_report  # noqa: E402


OUTPUT = Path("artifacts/instagram-reel-insights.json")
METRICS = (
    "views",
    "reach",
    "likes",
    "comments",
    "shares",
    "saved",
    "total_interactions",
    "ig_reels_avg_watch_time",
    "ig_reels_video_view_total_time",
)
METRIC_MAP = {
    "views": "views",
    "reach": "reach",
    "likes": "likes",
    "comments": "comments",
    "shares": "shares",
    "saved": "saves",
    "total_interactions": "total_interactions",
    "ig_reels_avg_watch_time": "average_watch_time_ms",
    "ig_reels_video_view_total_time": "total_watch_time_ms",
}


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la configuración protegida: {name}")
    return value


def graph_get(
    graph_base: str,
    path: str,
    params: dict[str, str],
) -> dict[str, Any]:
    url = f"{graph_base}/{path.lstrip('/')}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", {})
            message = detail.get("message") or f"HTTP {exc.code}"
        except json.JSONDecodeError:
            message = f"HTTP {exc.code}"
        raise RuntimeError(message) from exc
    if "error" in payload:
        raise RuntimeError(payload["error"].get("message", "Error de Instagram"))
    return payload


def find_media(
    graph_base: str,
    ig_user_id: str,
    token: str,
    permalink: str,
) -> dict[str, Any]:
    requested_media_id = os.environ.get("IG_MEDIA_ID", "").strip()
    if requested_media_id:
        return graph_get(
            graph_base,
            requested_media_id,
            {
                "fields": "id,permalink,timestamp,media_type,media_product_type",
                "access_token": token,
            },
        )

    payload = graph_get(
        graph_base,
        f"{ig_user_id}/media",
        {
            "fields": "id,permalink,timestamp,media_type,media_product_type",
            "limit": "100",
            "access_token": token,
        },
    )
    expected = permalink.rstrip("/")
    for media in payload.get("data", []):
        if str(media.get("permalink", "")).rstrip("/") == expected:
            return media
    raise RuntimeError("No se encontró el Reel solicitado entre los medios recientes")


def metric_value(payload: dict[str, Any]) -> float | int | None:
    rows = payload.get("data", [])
    if not rows:
        return None
    values = rows[0].get("values", [])
    value = values[-1].get("value") if values else rows[0].get("value")
    return value if isinstance(value, (int, float)) else None


def collect_metrics(
    graph_base: str,
    media_id: str,
    token: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    collected: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for metric in METRICS:
        try:
            payload = graph_get(
                graph_base,
                f"{media_id}/insights",
                {"metric": metric, "access_token": token},
            )
            value = metric_value(payload)
            if value is None:
                unavailable[metric] = "sin_valor"
            else:
                collected[METRIC_MAP[metric]] = value
        except RuntimeError as exc:
            unavailable[metric] = str(exc).replace(token, "***")[:180]
    return collected, unavailable


def write_output(payload: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    token = required_env("IG_ACCESS_TOKEN")
    ig_user_id = required_env("IG_USER_ID")
    permalink = required_env("IG_REEL_PERMALINK")
    graph_version = os.environ.get("GRAPH_VERSION", "v25.0")
    graph_base = f"https://graph.facebook.com/{graph_version}"

    media = find_media(graph_base, ig_user_id, token, permalink)
    metrics, unavailable = collect_metrics(
        graph_base,
        str(media["id"]),
        token,
    )
    baseline_views = int(os.environ.get("BASELINE_VIEWS", "0") or 0)
    baseline = {"views": baseline_views} if baseline_views else None
    report = {
        "schema_version": "instagram_reel_insights_v1",
        "mode": "read_only",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "media": media,
        "metrics": metrics,
        "unavailable_metrics": unavailable,
        "learning_report": build_reel_learning_report(metrics, baseline),
        "token_included": False,
    }
    write_output(report)
    print(f"Métricas guardadas en {OUTPUT}")


if __name__ == "__main__":
    main()
