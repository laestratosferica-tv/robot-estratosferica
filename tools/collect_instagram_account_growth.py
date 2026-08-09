#!/usr/bin/env python3
"""Collect a read-only weekly Instagram account snapshot.

The resulting readiness score is an internal planning signal. It is not an
Amazon eligibility score and never triggers an application or publication.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.collect_instagram_reel_insights import (  # noqa: E402
    graph_get,
    required_env,
)


OUTPUT_DIR = Path("artifacts/instagram-account-growth")
LATEST_OUTPUT = OUTPUT_DIR / "latest.json"
ACCOUNT_FIELDS = (
    "id,username,name,biography,followers_count,follows_count,media_count"
)
WEEKLY_METRICS = (
    "reach",
    "views",
    "total_interactions",
    "accounts_engaged",
    "profile_views",
    "website_clicks",
)


def metric_total(payload: dict[str, Any]) -> float | int | None:
    rows = payload.get("data", [])
    if not rows:
        return None
    values = rows[0].get("values", [])
    numeric = [item.get("value") for item in values]
    numeric = [value for value in numeric if isinstance(value, (int, float))]
    if numeric:
        return sum(numeric)
    value = rows[0].get("total_value", {}).get("value")
    return value if isinstance(value, (int, float)) else None


def collect_weekly_insights(
    graph_base: str,
    ig_user_id: str,
    token: str,
    since: int,
    until: int,
) -> tuple[dict[str, float | int], dict[str, str]]:
    collected: dict[str, float | int] = {}
    unavailable: dict[str, str] = {}
    for metric in WEEKLY_METRICS:
        try:
            payload = graph_get(
                graph_base,
                f"{ig_user_id}/insights",
                {
                    "metric": metric,
                    "period": "day",
                    "metric_type": "total_value",
                    "since": str(since),
                    "until": str(until),
                    "access_token": token,
                },
            )
            value = metric_total(payload)
            if value is None:
                unavailable[metric] = "sin_valor"
            else:
                collected[metric] = value
        except RuntimeError as exc:
            unavailable[metric] = str(exc).replace(token, "***")[:180]
    return collected, unavailable


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def build_internal_readiness(
    account: dict[str, Any],
    insights: dict[str, float | int],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a transparent, non-Amazon readiness score for planning."""
    previous = previous or {}
    previous_account = previous.get("account", {})
    followers = int(account.get("followers_count") or 0)
    previous_followers = int(previous_account.get("followers_count") or 0)
    media_count = int(account.get("media_count") or 0)
    previous_media = int(previous_account.get("media_count") or 0)
    reach = float(insights.get("reach") or insights.get("views") or 0)
    interactions = float(insights.get("total_interactions") or 0)
    follower_growth = followers - previous_followers if previous_followers else None
    weekly_posts = media_count - previous_media if previous_media else None
    engagement_rate = _ratio(interactions, reach)

    targets = {
        "followers": int(os.environ.get("INTERNAL_FOLLOWER_TARGET", "2000")),
        "weekly_posts": int(os.environ.get("INTERNAL_WEEKLY_POST_TARGET", "3")),
        "engagement_rate": float(
            os.environ.get("INTERNAL_ENGAGEMENT_TARGET", "0.03")
        ),
        "weekly_follower_growth_rate": float(
            os.environ.get("INTERNAL_GROWTH_TARGET", "0.01")
        ),
    }
    growth_rate = (
        _ratio(follower_growth or 0, previous_followers)
        if previous_followers
        else None
    )
    component_scores = {
        "audience": min(1.0, _ratio(followers, targets["followers"])),
        "engagement": min(
            1.0,
            _ratio(engagement_rate, targets["engagement_rate"]),
        ),
        "consistency": (
            min(1.0, _ratio(weekly_posts, targets["weekly_posts"]))
            if weekly_posts is not None
            else 0.0
        ),
        "growth": (
            min(1.0, _ratio(growth_rate, targets["weekly_follower_growth_rate"]))
            if growth_rate is not None
            else 0.0
        ),
    }
    score = round(
        100
        * (
            component_scores["audience"] * 0.30
            + component_scores["engagement"] * 0.35
            + component_scores["consistency"] * 0.20
            + component_scores["growth"] * 0.15
        ),
        1,
    )
    has_baseline = previous_followers > 0 and previous_media > 0
    data_complete = reach > 0 and "total_interactions" in insights and has_baseline
    return {
        "score": score,
        "label": "internal_planning_signal_only",
        "amazon_threshold_claimed": False,
        "automatic_reapplication_enabled": False,
        "requires_human_review": True,
        "data_quality": "complete" if data_complete else "partial",
        "targets": targets,
        "component_scores": component_scores,
        "observed": {
            "followers": followers,
            "follower_growth": follower_growth,
            "weekly_follower_growth_rate": growth_rate,
            "weekly_posts": weekly_posts,
            "reach": int(reach),
            "total_interactions": int(interactions),
            "engagement_rate": engagement_rate,
        },
        "recommendation": (
            "review_reapplication_manually"
            if data_complete and score >= 80
            else "continue_growth_experiments"
        ),
    }


def load_previous() -> dict[str, Any] | None:
    if not LATEST_OUTPUT.exists():
        return None
    try:
        return json.loads(LATEST_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_outputs(payload: dict[str, Any], collected_at: datetime) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    dated_output = OUTPUT_DIR / f"{collected_at.date().isoformat()}.json"
    dated_output.write_text(serialized, encoding="utf-8")
    LATEST_OUTPUT.write_text(serialized, encoding="utf-8")


def main() -> None:
    token = required_env("IG_ACCESS_TOKEN")
    ig_user_id = required_env("IG_USER_ID")
    graph_version = os.environ.get("GRAPH_VERSION", "v25.0")
    graph_base = f"https://graph.facebook.com/{graph_version}"
    now = datetime.now(timezone.utc)
    since = int((now - timedelta(days=7)).timestamp())
    until = int(now.timestamp())

    account = graph_get(
        graph_base,
        ig_user_id,
        {"fields": ACCOUNT_FIELDS, "access_token": token},
    )
    insights, unavailable = collect_weekly_insights(
        graph_base,
        ig_user_id,
        token,
        since,
        until,
    )
    previous = load_previous()
    payload = {
        "schema_version": "instagram_account_growth_v1",
        "mode": "read_only",
        "collected_at": now.isoformat(),
        "window": {"days": 7, "since": since, "until": until},
        "account": account,
        "weekly_insights": insights,
        "unavailable_metrics": unavailable,
        "internal_readiness": build_internal_readiness(
            account,
            insights,
            previous,
        ),
        "token_included": False,
    }
    write_outputs(payload, now)
    print(f"Seguimiento semanal guardado en {LATEST_OUTPUT}")


if __name__ == "__main__":
    main()
