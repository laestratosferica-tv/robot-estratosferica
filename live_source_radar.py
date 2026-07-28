from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from urllib.error import HTTPError, URLError
from datetime import date
from pathlib import Path
from time import struct_time
from typing import Any, Callable, Mapping

from media_factory.radar import (
    RadarRejected,
    load_source_registry,
    normalize_story,
)
from media_factory.strategy import (
    classify_candidate,
    load_content_strategy,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "config" / "sources_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "live_candidates.json"
DEFAULT_REPORT = ROOT / "artifacts" / "live-radar-report.json"
DEFAULT_STRATEGY = ROOT / "config" / "content_strategy_v1.json"


class LiveRadarError(RuntimeError):
    pass


NETWORK_ERRORS = (ConnectionError, HTTPError, TimeoutError, URLError)


POSITIVE_DISCOVERY_TERMS = {
    " ia ": 3,
    "artificial intelligence": 3,
    "inteligencia artificial": 3,
    "gemini": 3,
    "game pass": 3,
    "backward compatibility": 3,
    "cloud gaming": 3,
    "esports": 3,
    "developer": 2,
    "technology": 2,
    "partnership": 2,
    "ecosystem": 2,
    "xbox on pc": 2,
    "xbox en pc": 2,
    "agent": 2,
    "halo": 1,
}

NEGATIVE_DISCOVERY_TERMS = {
    "giveaway": -4,
    "win a ": -4,
    "next week on xbox": -5,
    "new games for": -2,
    "free play days": -3,
    "tips": -2,
    "podcast": -1,
    "launches today": -1,
}


def _entry_value(entry: Any, key: str, default: Any = "") -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _plain_text(value: Any) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", str(value or ""))
    return " ".join(html.unescape(without_tags).split())


def _clean_feed_summary(value: Any) -> str:
    summary = _plain_text(value)
    boilerplate_patterns = (
        r"\s+La entrada .+? se public[oó] primero en .+?\.?$",
        r"\s+The post .+? appeared first on .+?\.?$",
        r"\s+(?:Leer m[aá]s|Read|Continue reading)\.?$",
    )
    for pattern in boilerplate_patterns:
        summary = re.sub(pattern, "", summary, flags=re.IGNORECASE)
    return summary.strip()


def _published_date(entry: Any) -> str:
    parsed = _entry_value(entry, "published_parsed") or _entry_value(
        entry,
        "updated_parsed",
    )
    if not isinstance(parsed, struct_time) and not (
        isinstance(parsed, tuple) and len(parsed) >= 3
    ):
        raise LiveRadarError("entry_missing_machine_readable_date")
    return date(int(parsed[0]), int(parsed[1]), int(parsed[2])).isoformat()


def _discovery_priority(title: str, summary: str) -> tuple[int, list[str]]:
    text = f" {title} {summary} ".casefold()
    matched: list[str] = []
    priority = 0
    for term, weight in {
        **POSITIVE_DISCOVERY_TERMS,
        **NEGATIVE_DISCOVERY_TERMS,
    }.items():
        if term in text:
            priority += weight
            matched.append(term)
    return max(-10, min(10, priority)), matched


def _signals_for(
    source: Mapping[str, Any],
    discovery_priority: int,
) -> dict[str, float]:
    primary = source.get("tier") == "primary"
    positive = max(0, discovery_priority)
    return {
        "latam_relevance": 0.65 if positive else 0.45,
        "explanatory_value": min(0.90, 0.62 + positive * 0.05),
        "angle_originality": min(0.85, 0.60 + positive * 0.04),
        "verifiability": 1.0 if primary else 0.75,
        "conversation_potential": min(0.85, 0.60 + positive * 0.03),
        "commercial_potential": min(0.85, 0.55 + positive * 0.04),
    }


def _candidate_id(source_id: str, source_url: str) -> str:
    raw = f"{source_id}\0{source_url}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_feeds(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    feeds = [
        source
        for source in registry.get("sources", [])
        if source.get("enabled") and source.get("feed_url")
    ]
    if not feeds:
        raise LiveRadarError("No enabled live feeds in the source registry")
    return feeds


def _default_parser(feed_url: str) -> Any:
    try:
        import feedparser
    except ImportError as exc:
        raise LiveRadarError(
            "feedparser is required for a live radar execution"
        ) from exc
    return feedparser.parse(feed_url)


def _source_response(parsed: Any) -> tuple[str, str | None]:
    """Classify a feed response without treating an empty feed as a failure."""
    exception = _entry_value(parsed, "bozo_exception", None)
    if isinstance(exception, NETWORK_ERRORS):
        return "inaccessible", type(exception).__name__
    if bool(_entry_value(parsed, "bozo", False)) and not _entry_value(
        parsed, "entries", []
    ):
        return "error", type(exception).__name__ if exception else "parse_error"
    return "accessible", None


def _source_result(
    source: Mapping[str, Any],
    feed_url: str,
    status: str,
    *,
    error_type: str | None = None,
    entries_seen: int = 0,
    accepted: int = 0,
    rejected: int = 0,
) -> dict[str, Any]:
    return {
        "source_id": source["id"],
        "feed_url": feed_url,
        "status": status,
        "error_type": error_type,
        "entries_seen": entries_seen,
        "accepted": accepted,
        "rejected": rejected,
        "network_mode": "rss_read_only",
    }


def collect_live_candidates(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    max_per_source: int = 10,
    max_candidates: int = 20,
    today: date | None = None,
    parser: Callable[[str], Any] | None = None,
    strategy_path: str | Path = DEFAULT_STRATEGY,
) -> dict[str, Any]:
    if max_per_source < 1 or max_candidates < 1:
        raise LiveRadarError("Radar limits must be positive")

    current_date = today or date.today()
    parser = parser or _default_parser
    registry = load_source_registry(registry_path)
    strategy = load_content_strategy(strategy_path)
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    source_results: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    for source in _source_feeds(registry):
        feed_url = str(source["feed_url"])
        try:
            parsed = parser(feed_url)
        except NETWORK_ERRORS as exc:
            source_results.append(
                _source_result(
                    source,
                    feed_url,
                    "inaccessible",
                    error_type=type(exc).__name__,
                )
            )
            continue
        except Exception as exc:
            source_results.append(
                _source_result(
                    source,
                    feed_url,
                    "error",
                    error_type=type(exc).__name__,
                )
            )
            continue

        source_status, error_type = _source_response(parsed)
        entries = list(_entry_value(parsed, "entries", []))
        if source_status != "accessible":
            source_results.append(
                _source_result(
                    source,
                    feed_url,
                    source_status,
                    error_type=error_type,
                )
            )
            continue
        accepted_for_source = 0
        rejected_for_source = 0

        for entry in entries[:max_per_source]:
            try:
                source_url = str(_entry_value(entry, "link", "")).strip()
                if not source_url or source_url in seen_urls:
                    raise LiveRadarError("missing_or_duplicate_source_url")
                title = _plain_text(_entry_value(entry, "title"))
                summary = _clean_feed_summary(
                    _entry_value(
                        entry,
                        "summary",
                        _entry_value(entry, "description", ""),
                    )
                )
                discovery_priority, discovery_reasons = _discovery_priority(
                    title,
                    summary,
                )
                raw = {
                    "candidate_id": _candidate_id(
                        str(source["id"]),
                        source_url,
                    ),
                    "title": title,
                    "summary": summary,
                    "source_url": source_url,
                    "source_id": str(source["id"]),
                    "published_at": _published_date(entry),
                    "territory": str(source["default_territory"]),
                    "region": (
                        "latam"
                        if source.get("language") == "es-419"
                        else "global_relevant_to_latam"
                    ),
                    "is_duplicate": False,
                    "is_verified": not bool(
                        source.get("requires_corroboration", False)
                    ),
                    "has_media_rights": True,
                    "claims_supported": True,
                    "signals": _signals_for(source, discovery_priority),
                    "discovery_priority": discovery_priority,
                    "discovery_reasons": discovery_reasons,
                }
                if not raw["title"]:
                    raise LiveRadarError("entry_missing_title")
                candidate = normalize_story(
                    raw,
                    registry,
                    today=current_date,
                )
                raw["strategic_classification"] = classify_candidate(
                    raw,
                    strategy,
                )
            except (LiveRadarError, RadarRejected) as exc:
                reason = str(exc)
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                rejected_for_source += 1
                continue

            seen_urls.add(candidate.source_url)
            candidates.append(raw)
            accepted_for_source += 1

        source_results.append(
            _source_result(
                source,
                feed_url,
                "accessible_with_entries" if entries else "accessible_empty",
                entries_seen=min(len(entries), max_per_source),
                accepted=accepted_for_source,
                rejected=rejected_for_source,
            )
        )

    candidates.sort(
        key=lambda item: (
            item["discovery_priority"],
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

    accessible_statuses = {"accessible_with_entries", "accessible_empty"}
    has_accessible_source = any(
        source["status"] in accessible_statuses for source in source_results
    )
    has_failed_source = any(
        source["status"] in {"inaccessible", "error"}
        for source in source_results
    )
    all_sources_inaccessible = bool(source_results) and all(
        source["status"] == "inaccessible" for source in source_results
    )
    report = {
        "healthy": has_accessible_source,
        "status": (
            "network_failure" if all_sources_inaccessible
            else "partial" if has_accessible_source and has_failed_source
            else "ok" if has_accessible_source
            else "source_failure"
        ),
        "mode": "live_rss_read_only",
        "scan_date": current_date.isoformat(),
        "sources_scanned": len(source_results),
        "candidate_count": len(candidates),
        "rejection_counts": rejection_counts,
        "sources": source_results,
        "publishing_attempted": False,
        "external_writes_attempted": False,
        "paid_generation_attempted": False,
        "media_download_attempted": False,
        "measured_cost_usd": 0.0,
    }
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect current candidates from approved RSS feeds."
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--max-per-source", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()
    report = collect_live_candidates(
        registry_path=args.registry,
        output_path=args.output,
        report_path=args.report,
        max_per_source=args.max_per_source,
        max_candidates=args.max_candidates,
    )
    print(
        "Radar en vivo terminado: "
        f"{report['sources_scanned']} fuentes, "
        f"{report['candidate_count']} candidatos, "
        "0 publicaciones, USD 0.00"
    )
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
