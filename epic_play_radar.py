from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "epic_play_candidates.json"
DEFAULT_REPORT = ROOT / "artifacts" / "epic-play-radar-report.json"
DEFAULT_LIVE_CONFIG = ROOT / "config" / "epic_play_sources_v1.json"

SUPPORTED_PLATFORMS = {"twitch", "youtube"}
COMPETITIVE_TERMS = {
    "ace": 18,
    "clutch": 18,
    "pentakill": 18,
    "world record": 16,
    "récord mundial": 16,
    "final": 10,
    "semifinal": 8,
    "tournament": 8,
    "torneo": 8,
    "ranked": 6,
    "competitivo": 6,
    "highlights": 5,
    "mejores jugadas": 5,
}
COMMUNITY_TERMS = {
    "reaction": 4,
    "reacción": 4,
    "increíble": 5,
    "insane": 5,
    "impossible": 5,
    "imposible": 5,
}


class EpicPlayRadarError(RuntimeError):
    pass


def _value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _published_date(value: Any) -> date:
    raw = _clean_text(value)
    if not raw:
        raise EpicPlayRadarError("candidate_missing_published_at")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EpicPlayRadarError("candidate_invalid_published_at") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.date()


def _candidate_id(platform: str, external_id: str, source_url: str) -> str:
    raw = f"{platform}\0{external_id}\0{source_url}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _engagement_score(views: int) -> int:
    if views >= 1_000_000:
        return 35
    if views >= 250_000:
        return 30
    if views >= 50_000:
        return 24
    if views >= 10_000:
        return 18
    if views >= 2_500:
        return 12
    if views >= 500:
        return 6
    return 0


def _freshness_score(published: date, today: date) -> int:
    age_days = (today - published).days
    if age_days < 0:
        raise EpicPlayRadarError("candidate_published_in_future")
    if age_days <= 1:
        return 20
    if age_days <= 3:
        return 16
    if age_days <= 7:
        return 10
    if age_days <= 14:
        return 4
    raise EpicPlayRadarError("candidate_too_old")


def _text_signals(title: str, description: str) -> tuple[int, list[str]]:
    text = f"{title} {description}".casefold()
    score = 0
    matched: list[str] = []
    for term, weight in {**COMPETITIVE_TERMS, **COMMUNITY_TERMS}.items():
        if term in text:
            score += weight
            matched.append(term)
    return min(30, score), matched


def _normalize_candidate(
    item: Any,
    *,
    platform: str,
    today: date,
) -> dict[str, Any]:
    if platform not in SUPPORTED_PLATFORMS:
        raise EpicPlayRadarError("unsupported_platform")

    external_id = _clean_text(
        _value(item, "external_id", _value(item, "id"))
    )
    source_url = _clean_text(
        _value(item, "source_url", _value(item, "url"))
    )
    creator_name = _clean_text(
        _value(
            item,
            "creator_name",
            _value(item, "broadcaster_name", _value(item, "channel")),
        )
    )
    title = _clean_text(_value(item, "title"))
    description = _clean_text(_value(item, "description"))
    game = _clean_text(
        _value(item, "game", _value(item, "game_name", "Gaming"))
    )
    published = _published_date(
        _value(item, "published_at", _value(item, "created_at"))
    )

    if not external_id:
        raise EpicPlayRadarError("candidate_missing_external_id")
    if not source_url.startswith("https://"):
        raise EpicPlayRadarError("candidate_invalid_source_url")
    if not creator_name:
        raise EpicPlayRadarError("candidate_missing_creator")
    if not title:
        raise EpicPlayRadarError("candidate_missing_title")

    try:
        views = max(0, int(_value(item, "view_count", 0) or 0))
    except (TypeError, ValueError) as exc:
        raise EpicPlayRadarError("candidate_invalid_view_count") from exc

    text_score, matched_terms = _text_signals(title, description)
    score = (
        _engagement_score(views)
        + _freshness_score(published, today)
        + text_score
    )

    return {
        "candidate_id": _candidate_id(platform, external_id, source_url),
        "platform": platform,
        "external_id": external_id,
        "title": title,
        "description": description,
        "creator_name": creator_name,
        "creator_url": _clean_text(_value(item, "creator_url")),
        "game": game,
        "source_url": source_url,
        "published_at": published.isoformat(),
        "view_count": views,
        "discovery_score": score,
        "discovery_reasons": matched_terms,
        "editorial_lane": "epic_plays_and_creators",
        "commercial_lanes": [
            "weekly_play_sponsorship",
            "creator_partnership",
            "community_challenge",
        ],
        "rights": {
            "state": "link_only_unverified",
            "owner": creator_name,
            "reuse_allowed": False,
            "download_allowed": False,
            "republication_allowed": False,
            "permission_evidence": None,
            "required_action": "request_permission_or_use_official_embed",
        },
        "workflow": {
            "status": "awaiting_editorial_and_rights_review",
            "automatic_publish_allowed": False,
            "human_approval_required": True,
        },
    }


def collect_epic_play_candidates(
    *,
    providers: Mapping[str, Callable[[], Iterable[Any]]],
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    today: date | None = None,
    max_candidates: int = 25,
) -> dict[str, Any]:
    if max_candidates < 1:
        raise EpicPlayRadarError("max_candidates_must_be_positive")

    current_date = today or date.today()
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    rejection_counts: dict[str, int] = {}
    provider_results: list[dict[str, Any]] = []

    for platform, provider in providers.items():
        accepted = 0
        rejected = 0
        items = list(provider())
        for item in items:
            try:
                candidate = _normalize_candidate(
                    item,
                    platform=platform,
                    today=current_date,
                )
                key = (platform, candidate["external_id"])
                if key in seen:
                    raise EpicPlayRadarError("duplicate_candidate")
            except EpicPlayRadarError as exc:
                reason = str(exc)
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                rejected += 1
                continue

            seen.add(key)
            candidates.append(candidate)
            accepted += 1

        provider_results.append(
            {
                "platform": platform,
                "items_seen": len(items),
                "accepted": accepted,
                "rejected": rejected,
                "network_mode": "metadata_discovery_only",
            }
        )

    candidates.sort(
        key=lambda item: (
            item["discovery_score"],
            item["view_count"],
            item["published_at"],
            item["candidate_id"],
        ),
        reverse=True,
    )
    candidates = candidates[:max_candidates]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "healthy": True,
        "mode": "epic_play_metadata_discovery_only",
        "scan_date": current_date.isoformat(),
        "providers_scanned": len(provider_results),
        "candidate_count": len(candidates),
        "rejection_counts": rejection_counts,
        "providers": provider_results,
        "publishing_attempted": False,
        "external_writes_attempted": False,
        "paid_generation_attempted": False,
        "media_download_attempted": False,
        "rights_assumed": False,
        "measured_cost_usd": 0.0,
    }
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _load_fixture(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise EpicPlayRadarError("fixture_must_be_a_list")
    return payload


def _load_live_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EpicPlayRadarError("live_config_must_be_an_object")
    if payload.get("mode") != "metadata_discovery_only":
        raise EpicPlayRadarError("live_config_must_be_metadata_only")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score Twitch and YouTube metadata without downloading or publishing."
        )
    )
    parser.add_argument("--twitch-fixture")
    parser.add_argument("--youtube-fixture")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use official read-only metadata APIs.",
    )
    parser.add_argument("--live-config", default=str(DEFAULT_LIVE_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    providers: dict[str, Callable[[], Iterable[Any]]] = {}
    if args.twitch_fixture:
        providers["twitch"] = lambda: _load_fixture(args.twitch_fixture)
    if args.youtube_fixture:
        providers["youtube"] = lambda: _load_fixture(args.youtube_fixture)
    if args.live:
        if providers:
            raise EpicPlayRadarError("fixtures_and_live_mode_cannot_be_combined")
        from epic_play_providers import (
            twitch_clip_provider,
            youtube_video_provider,
        )

        live_config = _load_live_config(args.live_config)
        twitch = live_config.get("twitch") or {}
        youtube = live_config.get("youtube") or {}
        providers["twitch"] = lambda: twitch_clip_provider(
            game_names=twitch.get("games") or [],
            clips_per_game=int(twitch.get("clips_per_game", 10)),
            lookback_hours=int(twitch.get("lookback_hours", 72)),
        )
        providers["youtube"] = lambda: youtube_video_provider(
            queries=youtube.get("queries") or [],
            max_queries=int(youtube.get("max_queries_per_run", 2)),
            results_per_query=int(youtube.get("results_per_query", 10)),
            lookback_hours=int(youtube.get("lookback_hours", 72)),
        )
    if not providers:
        raise EpicPlayRadarError(
            "Provide metadata fixtures or explicitly pass --live."
        )

    report = collect_epic_play_candidates(
        providers=providers,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
