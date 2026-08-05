#!/usr/bin/env python3
"""Assess sustained Instagram readiness without applying to Amazon."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any


SNAPSHOT_DIR = Path("artifacts/instagram-account-growth")
OUTPUT = SNAPSHOT_DIR / "reapplication-readiness.json"
WINDOW_WEEKS = 4


def load_snapshots(snapshot_dir: Path = SNAPSHOT_DIR) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for path in sorted(snapshot_dir.glob("20??-??-??.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "instagram_account_growth_v1":
            snapshots.append(payload)
    return snapshots


def assess_sustained_readiness(
    snapshots: list[dict[str, Any]],
    window_weeks: int = WINDOW_WEEKS,
) -> dict[str, Any]:
    window = snapshots[-window_weeks:]
    readiness = [item.get("internal_readiness", {}) for item in window]
    observations = [item.get("observed", {}) for item in readiness]
    qualities = [item.get("data_quality") for item in readiness]
    targets = readiness[-1].get("targets", {}) if readiness else {}
    follower_target = int(targets.get("followers", 2000))
    post_target = int(targets.get("weekly_posts", 3))
    engagement_target = float(targets.get("engagement_rate", 0.03))

    weekly_posts = [item.get("weekly_posts") for item in observations]
    engagement_rates = [item.get("engagement_rate") for item in observations]
    growth_rates = [item.get("weekly_follower_growth_rate") for item in observations]
    latest_followers = int(observations[-1].get("followers") or 0) if observations else 0
    complete_window = len(window) == window_weeks

    checks = {
        "four_week_history": complete_window,
        "complete_weeks": complete_window and all(value == "complete" for value in qualities),
        "posting_consistency": complete_window
        and all(isinstance(value, (int, float)) and value >= post_target for value in weekly_posts),
        "engagement_consistency": complete_window
        and sum(
            isinstance(value, (int, float)) and value >= engagement_target
            for value in engagement_rates
        )
        >= 3,
        "positive_growth_consistency": complete_window
        and sum(
            isinstance(value, (int, float)) and value > 0 for value in growth_rates
        )
        >= 3,
        "audience_target": latest_followers >= follower_target,
    }
    ready = all(checks.values())
    numeric_engagement = [
        float(value) for value in engagement_rates if isinstance(value, (int, float))
    ]
    numeric_growth = [
        float(value) for value in growth_rates if isinstance(value, (int, float))
    ]
    return {
        "schema_version": "amazon_influencer_reapplication_readiness_v1",
        "mode": "analysis_only",
        "weeks_required": window_weeks,
        "weeks_available": len(window),
        "checks": checks,
        "observed": {
            "latest_followers": latest_followers,
            "median_weekly_engagement_rate": (
                round(median(numeric_engagement), 6) if numeric_engagement else None
            ),
            "median_weekly_follower_growth_rate": (
                round(median(numeric_growth), 6) if numeric_growth else None
            ),
            "weekly_posts": weekly_posts,
        },
        "ready_for_human_reapplication_review": ready,
        "amazon_approval_guaranteed": False,
        "automatic_reapplication_enabled": False,
        "requires_human_review": True,
        "recommendation": (
            "review_reapplication_manually"
            if ready
            else "continue_growth_and_measurement"
        ),
    }


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    report = assess_sustained_readiness(load_snapshots())
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluación sostenible guardada en {OUTPUT}")


if __name__ == "__main__":
    main()
