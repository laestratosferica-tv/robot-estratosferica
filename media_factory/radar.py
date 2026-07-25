from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import Candidate


class RadarRejected(ValueError):
    pass


def load_source_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        registry = json.load(handle)
    if not registry.get("sources"):
        raise ValueError("El registro debe incluir fuentes")
    return registry


def _domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(
        host == domain.lower() or host.endswith(f".{domain.lower()}")
        for domain in allowed_domains
    )


def normalize_story(
    data: dict[str, Any],
    registry: dict[str, Any],
    today: date | None = None,
) -> Candidate:
    source_id = str(data.get("source_id", "")).strip()
    sources = {
        source["id"]: source
        for source in registry["sources"]
        if source.get("enabled", False)
    }
    if source_id not in sources:
        raise RadarRejected("source_not_allowed")
    source = sources[source_id]
    source_url = str(data.get("source_url", "")).strip()
    if not _domain_allowed(source_url, source.get("allowed_domains", [])):
        raise RadarRejected("source_domain_mismatch")
    published_raw = str(data.get("published_at", "")).strip()
    try:
        published = date.fromisoformat(published_raw)
    except ValueError as exc:
        raise RadarRejected("invalid_published_at") from exc
    current_date = today or date.today()
    age_days = (current_date - published).days
    if age_days < -1:
        raise RadarRejected("future_story")
    if age_days > int(registry.get("max_age_days", 14)):
        raise RadarRejected("stale_story")
    title = str(data.get("title", "")).strip()
    summary = str(data.get("summary", "")).strip()
    combined = f"{title} {summary}".lower()
    if any(
        keyword.lower() in combined
        for keyword in registry.get("disallowed_keywords", [])
    ):
        raise RadarRejected("blocked_topic")
    territory = str(data.get("territory", "")).strip()
    if territory not in source.get("territories", []):
        raise RadarRejected("source_territory_mismatch")
    return Candidate.from_dict(data)
