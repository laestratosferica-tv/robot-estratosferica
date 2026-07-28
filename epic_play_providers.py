from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping


TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_API = "https://api.twitch.tv/helix"
YOUTUBE_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"


class ProviderConfigurationError(RuntimeError):
    pass


def _network_session() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests_dependency_missing") from exc
    return requests


def _required_env(name: str, env: Mapping[str, str]) -> str:
    value = str(env.get(name, "")).strip()
    if not value:
        raise ProviderConfigurationError(f"missing_{name.lower()}")
    return value


def _json(response: Any) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("provider_response_must_be_object")
    return payload


def twitch_clip_provider(
    *,
    game_names: Iterable[str],
    env: Mapping[str, str] | None = None,
    session: Any = None,
    now: datetime | None = None,
    clips_per_game: int = 10,
    lookback_hours: int = 72,
) -> list[dict[str, Any]]:
    """Read current Twitch clip metadata. Never requests download endpoints."""
    runtime_env = env or os.environ
    client = session or _network_session()
    client_id = _required_env("TWITCH_CLIENT_ID", runtime_env)
    client_secret = _required_env("TWITCH_CLIENT_SECRET", runtime_env)
    token_payload = _json(
        client.post(
            TWITCH_TOKEN_URL,
            params={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=20,
        )
    )
    access_token = str(token_payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("twitch_missing_access_token")

    headers = {
        "Client-Id": client_id,
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "La-Estratosferica-Safe-Radar/1.0",
    }
    current = now or datetime.now(timezone.utc)
    started_at = (current - timedelta(hours=max(1, lookback_hours))).isoformat()
    started_at = started_at.replace("+00:00", "Z")
    limit = max(1, min(100, clips_per_game))
    results: list[dict[str, Any]] = []

    for game_name in list(game_names):
        games = _json(
            client.get(
                f"{TWITCH_API}/games",
                headers=headers,
                params={"name": game_name},
                timeout=20,
            )
        ).get("data", [])
        if not games:
            continue
        game_id = str(games[0].get("id", "")).strip()
        resolved_name = str(games[0].get("name", game_name)).strip()
        if not game_id:
            continue
        clips = _json(
            client.get(
                f"{TWITCH_API}/clips",
                headers=headers,
                params={
                    "game_id": game_id,
                    "started_at": started_at,
                    "first": limit,
                },
                timeout=20,
            )
        ).get("data", [])
        for clip in clips:
            results.append(
                {
                    "id": clip.get("id"),
                    "title": clip.get("title"),
                    "description": "",
                    "creator_name": (
                        clip.get("broadcaster_name") or clip.get("creator_name")
                    ),
                    "creator_url": (
                        f"https://www.twitch.tv/{clip.get('broadcaster_name')}"
                        if clip.get("broadcaster_name")
                        else ""
                    ),
                    "game_name": resolved_name,
                    "url": clip.get("url"),
                    "created_at": clip.get("created_at"),
                    "view_count": clip.get("view_count", 0),
                }
            )
    return results


def youtube_video_provider(
    *,
    queries: Iterable[str],
    env: Mapping[str, str] | None = None,
    session: Any = None,
    now: datetime | None = None,
    max_queries: int = 2,
    results_per_query: int = 10,
    lookback_hours: int = 72,
) -> list[dict[str, Any]]:
    """Read YouTube search/video metadata under a strict search-quota cap."""
    runtime_env = env or os.environ
    client = session or _network_session()
    client_id = _required_env("YOUTUBE_CLIENT_ID", runtime_env)
    client_secret = _required_env("YOUTUBE_CLIENT_SECRET", runtime_env)
    refresh_token = _required_env("YOUTUBE_REFRESH_TOKEN", runtime_env)
    token_payload = _json(
        client.post(
            YOUTUBE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=20,
        )
    )
    access_token = str(token_payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("youtube_missing_access_token")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "La-Estratosferica-Safe-Radar/1.0",
    }
    current = now or datetime.now(timezone.utc)
    published_after = (
        current - timedelta(hours=max(1, lookback_hours))
    ).isoformat().replace("+00:00", "Z")
    query_limit = max(1, min(4, max_queries))
    result_limit = max(1, min(50, results_per_query))
    snippets: dict[str, dict[str, Any]] = {}

    for query in list(queries)[:query_limit]:
        search = _json(
            client.get(
                f"{YOUTUBE_API}/search",
                headers=headers,
                params={
                    "part": "snippet",
                    "type": "video",
                    "order": "viewCount",
                    "q": query,
                    "publishedAfter": published_after,
                    "maxResults": result_limit,
                    "relevanceLanguage": "es",
                    "regionCode": "CO",
                },
                timeout=20,
            )
        )
        for item in search.get("items", []):
            video_id = str((item.get("id") or {}).get("videoId", "")).strip()
            if video_id:
                snippets.setdefault(video_id, item.get("snippet") or {})

    if not snippets:
        return []
    details = _json(
        client.get(
            f"{YOUTUBE_API}/videos",
            headers=headers,
            params={
                "part": "snippet,statistics",
                "id": ",".join(snippets),
                "maxResults": min(50, len(snippets)),
            },
            timeout=20,
        )
    )
    results: list[dict[str, Any]] = []
    for item in details.get("items", []):
        video_id = str(item.get("id", "")).strip()
        snippet = item.get("snippet") or snippets.get(video_id, {})
        statistics = item.get("statistics") or {}
        results.append(
            {
                "id": video_id,
                "title": snippet.get("title"),
                "description": snippet.get("description", ""),
                "channel": snippet.get("channelTitle"),
                "creator_url": (
                    f"https://www.youtube.com/channel/{snippet.get('channelId')}"
                    if snippet.get("channelId")
                    else ""
                ),
                "game": "Gaming",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "published_at": snippet.get("publishedAt"),
                "view_count": statistics.get("viewCount", 0),
            }
        )
    return results
