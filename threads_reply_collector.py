from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from community_commercial_radar import prioritize_interactions


DEFAULT_GRAPH_BASE = "https://graph.threads.net"
DEFAULT_FIELDS = "id,text,username,timestamp,permalink"


class ThreadsReadOnlyClient:
    """Minimal Threads client that only performs GET requests."""

    def __init__(
        self,
        access_token: str,
        *,
        graph_base: str = DEFAULT_GRAPH_BASE,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not access_token.strip():
            raise ValueError("THREADS_USER_ACCESS_TOKEN is required")
        self.access_token = access_token.strip()
        self.graph_base = graph_base.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get(self, path_or_url: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = (
            path_or_url
            if path_or_url.startswith("https://")
            else f"{self.graph_base}/{path_or_url.lstrip('/')}"
        )
        query = dict(params or {})
        query.setdefault("access_token", self.access_token)
        response = self.session.get(url, params=query, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Threads returned a non-object response")
        return payload

    def _pages(
        self,
        path: str,
        *,
        params: Dict[str, Any],
        max_pages: int,
    ) -> Iterable[Dict[str, Any]]:
        payload = self._get(path, params=params)
        pages = 0
        while True:
            pages += 1
            for item in payload.get("data", []):
                if isinstance(item, dict):
                    yield item
            next_url = payload.get("paging", {}).get("next")
            if not next_url or pages >= max_pages:
                break
            payload = self._get(str(next_url))

    def get_own_threads(self, *, limit: int = 10, max_pages: int = 2) -> List[Dict[str, Any]]:
        return list(
            self._pages(
                "me/threads",
                params={"fields": DEFAULT_FIELDS, "limit": max(1, min(limit, 100))},
                max_pages=max_pages,
            )
        )[:limit]

    def get_replies(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        max_pages: int = 3,
    ) -> List[Dict[str, Any]]:
        return list(
            self._pages(
                f"{thread_id}/replies",
                params={"fields": DEFAULT_FIELDS, "limit": max(1, min(limit, 100))},
                max_pages=max_pages,
            )
        )[:limit]


def collect_and_classify(
    client: ThreadsReadOnlyClient,
    *,
    thread_limit: int = 10,
    reply_limit: int = 100,
    minimum_commercial_score: int = 25,
    hot_score: int = 60,
) -> Dict[str, Any]:
    interactions: List[Dict[str, Any]] = []
    threads = client.get_own_threads(limit=thread_limit)

    for parent in threads:
        parent_id = str(parent.get("id", "")).strip()
        if not parent_id:
            continue
        for reply in client.get_replies(parent_id, limit=reply_limit):
            interactions.append(
                {
                    "platform": "threads",
                    "interaction_id": str(reply.get("id", "")),
                    "parent_thread_id": parent_id,
                    "parent_permalink": parent.get("permalink", ""),
                    "text": reply.get("text", ""),
                    "display_name": reply.get("username", ""),
                    "username": reply.get("username", ""),
                    "created_at": reply.get("timestamp", ""),
                    "permalink": reply.get("permalink", ""),
                }
            )

    queue = prioritize_interactions(
        interactions,
        minimum_commercial_score=minimum_commercial_score,
        hot_score=hot_score,
    )
    counts = {
        "commercial_lead": sum(item["classification"] == "commercial_lead" for item in queue),
        "community_signal": sum(item["classification"] == "community_signal" for item in queue),
        "risk_or_spam": sum(item["classification"] == "risk_or_spam" for item in queue),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "analysis_only",
        "outbound_actions_enabled": False,
        "threads_scanned": len(threads),
        "replies_scanned": len(interactions),
        "counts": counts,
        "queue": queue,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and classify Threads replies without writing.")
    parser.add_argument("--output", default="artifacts/threads-commercial-radar.json")
    parser.add_argument("--thread-limit", type=int, default=10)
    parser.add_argument("--reply-limit", type=int, default=100)
    args = parser.parse_args()

    token = os.getenv("THREADS_USER_ACCESS_TOKEN", "")
    client = ThreadsReadOnlyClient(
        token,
        graph_base=os.getenv("THREADS_GRAPH", DEFAULT_GRAPH_BASE),
    )
    report = collect_and_classify(
        client,
        thread_limit=max(1, args.thread_limit),
        reply_limit=max(1, args.reply_limit),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "threads_scanned": report["threads_scanned"],
                "replies_scanned": report["replies_scanned"],
                "counts": report["counts"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
