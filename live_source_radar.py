from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from datetime import date
from pathlib import Path
from time import struct_time
from typing import Any, Callable, Mapping

from media_factory.radar import (
    RadarRejected,
    load_source_registry,
    normalize_story,
)
from media_factory.editorial_quality import (
    normalize_editorial_text,
    substantive_summary_issue,
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
MAX_ARTICLE_BYTES = 512_000
MAX_EVIDENCE_CHARS = 700


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

ROUTINE_ROUNDUP_TERMS = {
    "la proxima semana en xbox",
    "la próxima semana en xbox",
    "next week on xbox",
}

TERRITORY_SIGNALS = {
    "gaming_esports": {
        "esports",
        "game pass",
        "gaming",
        "halo",
        "juego",
        "juegos",
        "jugabilidad",
        "jugador",
        "jugadores",
        "videojuego",
        "xbox",
    },
    "sport_technology_entertainment": {
        "deporte",
        "estadio",
        "futbol",
        "fútbol",
        "google earth",
        "partido",
        "sports",
        "stadium",
    },
    "brands_activations": {
        "activacion",
        "activación",
        "marca",
        "marketing",
        "patrocinio",
    },
    "ai_innovation_future": {
        "ai ",
        "gemini",
        "ia ",
        "inteligencia artificial",
        "machine learning",
    },
}


def _entry_value(entry: Any, key: str, default: Any = "") -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    return any(
        host == domain.casefold()
        or host.endswith(f".{domain.casefold()}")
        for domain in allowed_domains
    )


class _ArticleEvidenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.descriptions: list[str] = []
        self.paragraphs: list[str] = []
        self._paragraph_parts: list[str] | None = None
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {
            str(key).casefold(): str(value or "")
            for key, value in attrs
        }
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        if tag == "meta":
            marker = (
                attributes.get("name")
                or attributes.get("property")
                or ""
            ).casefold()
            if marker in {
                "description",
                "og:description",
                "twitter:description",
            }:
                content = _plain_text(attributes.get("content", ""))
                if content:
                    self.descriptions.append(content)
        if tag == "p" and not self._ignored_depth:
            self._paragraph_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._paragraph_parts is not None:
            paragraph = _plain_text(" ".join(self._paragraph_parts))
            if paragraph:
                self.paragraphs.append(paragraph)
            self._paragraph_parts = None
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._paragraph_parts is not None and not self._ignored_depth:
            self._paragraph_parts.append(data)


class _ApprovedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_domains: list[str]) -> None:
        super().__init__()
        self.allowed_domains = allowed_domains

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if not _domain_allowed(newurl, self.allowed_domains):
            raise LiveRadarError("article_redirect_domain_mismatch")
        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


def _plain_text(value: Any) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", str(value or ""))
    return " ".join(html.unescape(without_tags).split())


def _clean_feed_summary(value: Any, *, title: str = "") -> str:
    summary = _plain_text(value)
    boilerplate_patterns = (
        r"(?:^|\s+)La entrada .+? se public[oó] primero en .+?\s*\.?$",
        r"(?:^|\s+)The post .+? appeared first on .+?\s*\.?$",
        r"(?:^|\s+)(?:Leer m[aá]s|Read|Continue reading)\s*\.?$",
    )
    for pattern in boilerplate_patterns:
        summary = re.sub(pattern, "", summary, flags=re.IGNORECASE)
    return summary.strip()


def _routine_roundup_issue(title: str) -> str | None:
    normalized = _plain_text(title).casefold()
    if any(term in normalized for term in ROUTINE_ROUNDUP_TERMS):
        return "routine_release_roundup"
    return None


def _bounded_evidence(value: str) -> str:
    text = _plain_text(value)
    if len(text) <= MAX_EVIDENCE_CHARS:
        return text
    bounded = text[:MAX_EVIDENCE_CHARS]
    sentence_end = max(
        bounded.rfind(". "),
        bounded.rfind("? "),
        bounded.rfind("! "),
    )
    if sentence_end < 120:
        return ""
    return bounded[:sentence_end + 1].strip()


def _evidence_specificity_score(evidence: str) -> tuple[int, int]:
    """Prefer measurable findings and concrete changes over generic summaries."""
    normalized = normalize_editorial_text(evidence)
    tokens = set(normalized.split())
    numeric_facts = len(re.findall(r"\d+(?:[.,]\d+)?", evidence))
    concrete_terms = {
        "abarca",
        "abarcan",
        "agrega",
        "anade",
        "confirma",
        "incorpora",
        "incluye",
        "incluyen",
        "mejora",
        "permite",
        "revela",
        "suma",
    }
    concrete_score = len(tokens.intersection(concrete_terms))
    phrase_score = 2 if "se basa en" in normalized else 0
    return (
        numeric_facts * 5 + concrete_score * 3 + phrase_score,
        min(len(evidence), MAX_EVIDENCE_CHARS),
    )


def _article_evidence(html_document: str, title: str) -> str:
    parser = _ArticleEvidenceParser()
    parser.feed(str(html_document or ""))
    first_bounded_candidate = ""
    valid_candidates: list[str] = []
    for raw_candidate in [
        *parser.descriptions,
        *parser.paragraphs[:20],
    ]:
        evidence = _bounded_evidence(raw_candidate)
        first_bounded_candidate = first_bounded_candidate or evidence
        if not substantive_summary_issue(title, evidence):
            valid_candidates.append(evidence)
    if valid_candidates:
        return max(valid_candidates, key=_evidence_specificity_score)
    return first_bounded_candidate


def _default_article_fetcher(
    source_url: str,
    allowed_domains: list[str],
) -> str:
    request = Request(
        source_url,
        headers={
            "User-Agent": (
                "La-Estratosferica-Editorial-Radar/1.0 "
                "(read-only evidence verification)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    opener = build_opener(_ApprovedRedirectHandler(allowed_domains))
    with opener.open(request, timeout=6) as response:
        final_url = str(response.geturl())
        if not _domain_allowed(final_url, allowed_domains):
            raise LiveRadarError("article_redirect_domain_mismatch")
        content_type = str(response.headers.get("Content-Type", "")).casefold()
        if "html" not in content_type:
            raise LiveRadarError("article_not_html")
        payload = response.read(MAX_ARTICLE_BYTES + 1)
        if len(payload) > MAX_ARTICLE_BYTES:
            raise LiveRadarError("article_html_too_large")
        charset = response.headers.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")


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


def _territory_for(
    source: Mapping[str, Any],
    title: str,
    summary: str,
) -> str:
    """Resolve the story topic instead of inheriting the publisher's default."""
    text = f" {title} {summary} ".casefold()
    allowed = set(source.get("territories", []))
    ranked = [
        (
            sum(
                bool(re.search(
                    rf"(?<!\w){re.escape(term.strip())}(?!\w)",
                    text,
                ))
                for term in terms
            ),
            territory,
        )
        for territory, terms in TERRITORY_SIGNALS.items()
        if territory in allowed
    ]
    best_score, best_territory = max(
        ranked,
        default=(0, str(source["default_territory"])),
        key=lambda item: item[0],
    )
    if best_score:
        return best_territory
    if source.get("require_territory_signal", False):
        raise LiveRadarError("outside_editorial_territory")
    return str(source["default_territory"])


def _candidate_id(source_id: str, source_url: str) -> str:
    raw = f"{source_id}\0{source_url}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _story_key(
    source_id: str,
    title: str,
    published_at: str,
) -> tuple[str, str, str]:
    normalized = unicodedata.normalize("NFKD", title.casefold())
    without_marks = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    normalized_title = re.sub(r"[\W_]+", " ", without_marks).strip()
    return source_id, normalized_title, published_at


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
    network_mode: str = "rss_read_only",
) -> dict[str, Any]:
    return {
        "source_id": source["id"],
        "feed_url": feed_url,
        "status": status,
        "error_type": error_type,
        "entries_seen": entries_seen,
        "accepted": accepted,
        "rejected": rejected,
        "network_mode": network_mode,
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
    article_fetcher: Callable[[str, list[str]], str] | None = None,
    max_article_fetches_per_source: int = 2,
) -> dict[str, Any]:
    if (
        max_per_source < 1
        or max_candidates < 1
        or max_article_fetches_per_source < 0
    ):
        raise LiveRadarError("Radar limits must be positive")

    current_date = today or date.today()
    enrichment_enabled = article_fetcher is not None or parser is None
    parser = parser or _default_parser
    article_fetcher = article_fetcher or _default_article_fetcher
    registry = load_source_registry(registry_path)
    strategy = load_content_strategy(strategy_path)
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_story_keys: set[tuple[str, str, str]] = set()
    source_results: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    article_fetch_attempts = 0
    article_fetch_successes = 0

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
        article_fetches_for_source = 0

        for entry in entries[:max_per_source]:
            try:
                source_url = str(_entry_value(entry, "link", "")).strip()
                if not source_url or source_url in seen_urls:
                    raise LiveRadarError("missing_or_duplicate_source_url")
                allowed_domains = list(source.get("allowed_domains", []))
                if not _domain_allowed(source_url, allowed_domains):
                    raise LiveRadarError("source_domain_mismatch")
                title = _plain_text(_entry_value(entry, "title"))
                if not title:
                    raise LiveRadarError("entry_missing_title")
                roundup_issue = _routine_roundup_issue(title)
                if roundup_issue:
                    raise LiveRadarError(roundup_issue)
                published_at = _published_date(entry)
                story_key = _story_key(
                    str(source["id"]),
                    title,
                    published_at,
                )
                if story_key in seen_story_keys:
                    raise LiveRadarError("duplicate_story")
                summary = _clean_feed_summary(
                    _entry_value(
                        entry,
                        "summary",
                        _entry_value(entry, "description", ""),
                    ),
                    title=title,
                )
                summary_issue = substantive_summary_issue(title, summary)
                summary_origin = "rss"
                if (
                    summary_issue
                    and enrichment_enabled
                    and article_fetches_for_source
                    < max_article_fetches_per_source
                ):
                    article_fetches_for_source += 1
                    article_fetch_attempts += 1
                    try:
                        article_html = article_fetcher(
                            source_url,
                            allowed_domains,
                        )
                        enriched_summary = _article_evidence(
                            article_html,
                            title,
                        )
                    except Exception:
                        enriched_summary = ""
                    if enriched_summary:
                        summary = enriched_summary
                        summary_origin = "article_page"
                        summary_issue = substantive_summary_issue(
                            title,
                            summary,
                        )
                        if not summary_issue:
                            article_fetch_successes += 1
                if summary_issue:
                    raise LiveRadarError(summary_issue)
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
                    "summary_origin": summary_origin,
                    "source_url": source_url,
                    "source_id": str(source["id"]),
                    "published_at": published_at,
                    "territory": _territory_for(source, title, summary),
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
            seen_story_keys.add(story_key)
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
                network_mode=(
                    "rss_and_approved_article_read_only"
                    if article_fetches_for_source
                    else "rss_read_only"
                ),
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
        "mode": "live_rss_and_approved_article_read_only",
        "scan_date": current_date.isoformat(),
        "sources_scanned": len(source_results),
        "candidate_count": len(candidates),
        "article_fetch_attempts": article_fetch_attempts,
        "article_fetch_successes": article_fetch_successes,
        "article_fetch_limit_per_source": max_article_fetches_per_source,
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
