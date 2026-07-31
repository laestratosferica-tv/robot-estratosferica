from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping


GRAPH_API = "https://graph.facebook.com/v23.0"
THREADS_API = "https://graph.threads.net/v1.0"
GOOGLE_TOKEN_API = "https://oauth2.googleapis.com/token"

PLATFORM_REQUIREMENTS = {
    "threads": ("THREADS_USER_ACCESS_TOKEN", "THREADS_USER_ID"),
    "instagram": ("IG_ACCESS_TOKEN", "IG_USER_ID"),
    "facebook": ("FB_PAGE_ACCESS_TOKEN", "FB_PAGE_ID"),
    "youtube": (
        "YOUTUBE_CLIENT_ID",
        "YOUTUBE_CLIENT_SECRET",
        "YOUTUBE_REFRESH_TOKEN",
    ),
}


def _request(
    request: urllib.request.Request,
    *,
    opener: Any = urllib.request.urlopen,
) -> str:
    with opener(request, timeout=15) as response:
        response.read(1)
    return "valid"


def _request_json(
    request: urllib.request.Request,
    *,
    opener: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    with opener(request, timeout=15) as response:
        payload = json.loads(response.read(65536).decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _diagnose_facebook_publish_capability(
    *,
    access_token: str,
    opener: Any,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {access_token}"}
    permissions_request = urllib.request.Request(
        f"{GRAPH_API}/me/permissions",
        headers=headers,
        method="GET",
    )
    permissions_payload = _request_json(permissions_request, opener=opener)
    granted_permissions = sorted(
        {
            str(item.get("permission", "")).strip()
            for item in permissions_payload.get("data", [])
            if isinstance(item, dict)
            and item.get("status") == "granted"
            and str(item.get("permission", "")).strip()
        }
    )
    return {
        "pages_manage_posts_confirmed": "pages_manage_posts"
        in granted_permissions,
        "pages_read_engagement_confirmed": "pages_read_engagement"
        in granted_permissions,
        "reels_publish_permission_ready": "pages_manage_posts"
        in granted_permissions,
        "page_content_task_requires_business_suite_verification": True,
    }


def _diagnose_graph_platform(
    *,
    platform: str,
    identifier: str,
    access_token: str,
    opener: Any,
) -> str:
    base = THREADS_API if platform == "threads" else GRAPH_API
    fields = "id" if platform == "facebook" else "id,username"
    query = urllib.parse.urlencode({"fields": fields})
    request = urllib.request.Request(
        f"{base}/{urllib.parse.quote(identifier, safe='')}?{query}",
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    return _request(request, opener=opener)


def _diagnose_youtube(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    opener: Any,
) -> str:
    body = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GOOGLE_TOKEN_API,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    return _request(request, opener=opener)


def _redact(value: Any, secrets: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    safe = value
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        safe = safe.replace(secret, "***")
    safe = re.sub(
        r"(?i)(access[_ -]?token|client[_ -]?secret)([\"'=:\s]+)[^\s,\"'}]+",
        r"\1\2***",
        safe,
    )
    return safe[:500]


def _safe_error_details(
    error: Exception,
    *,
    secrets: list[str],
) -> dict[str, Any]:
    if isinstance(error, urllib.error.HTTPError):
        details: dict[str, Any] = {"http_status": error.code}
        try:
            payload = json.loads(error.read(65536).decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            payload = {}

        provider_error = payload.get("error", payload) if isinstance(payload, dict) else {}
        if isinstance(provider_error, str):
            details["provider_error"] = _redact(provider_error, secrets)
        elif isinstance(provider_error, dict):
            safe_fields = {
                "code": "provider_error_code",
                "error_subcode": "provider_error_subcode",
                "type": "provider_error_type",
                "message": "provider_message",
                "error": "provider_error",
                "error_description": "provider_message",
            }
            for source, target in safe_fields.items():
                value = provider_error.get(source)
                if value is not None:
                    details[target] = _redact(value, secrets)
        return details
    if isinstance(error, urllib.error.URLError):
        return {"network_reason": _redact(str(error.reason), secrets)}
    return {"exception_type": type(error).__name__}


def _safe_status(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        if error.code in {400, 401, 403}:
            return "provider_rejected"
        if error.code == 429:
            return "rate_limited"
        return "service_error"
    if isinstance(error, urllib.error.URLError):
        return "network_error"
    return "diagnostic_error"


def build_credential_diagnostic(
    environment: Mapping[str, str] | None = None,
    *,
    opener: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    secrets = [
        environment.get(name, "").strip()
        for names in PLATFORM_REQUIREMENTS.values()
        for name in names
    ]
    platforms: dict[str, Any] = {}

    for platform, names in PLATFORM_REQUIREMENTS.items():
        values = {name: environment.get(name, "").strip() for name in names}
        configured_count = sum(bool(value) for value in values.values())
        error_details: dict[str, Any] = {}
        publish_capability: dict[str, Any] = {}
        if configured_count == 0:
            status = "missing"
            checked = False
        elif configured_count != len(values):
            status = "incomplete"
            checked = False
        else:
            checked = True
            try:
                if platform == "youtube":
                    status = _diagnose_youtube(
                        client_id=values["YOUTUBE_CLIENT_ID"],
                        client_secret=values["YOUTUBE_CLIENT_SECRET"],
                        refresh_token=values["YOUTUBE_REFRESH_TOKEN"],
                        opener=opener,
                    )
                else:
                    token_name = (
                        "THREADS_USER_ACCESS_TOKEN"
                        if platform == "threads"
                        else "IG_ACCESS_TOKEN"
                        if platform == "instagram"
                        else "FB_PAGE_ACCESS_TOKEN"
                    )
                    id_name = (
                        "THREADS_USER_ID"
                        if platform == "threads"
                        else "IG_USER_ID"
                        if platform == "instagram"
                        else "FB_PAGE_ID"
                    )
                    status = _diagnose_graph_platform(
                        platform=platform,
                        identifier=values[id_name],
                        access_token=values[token_name],
                        opener=opener,
                    )
                    if platform == "facebook":
                        publish_capability = _diagnose_facebook_publish_capability(
                            access_token=values[token_name],
                            opener=opener,
                        )
            except Exception as error:
                status = _safe_status(error)
                error_details = _safe_error_details(error, secrets=secrets)

        platforms[platform] = {
            "status": status,
            "configured_count": configured_count,
            "required_count": len(names),
            "live_readonly_check_performed": checked,
            "secret_values_exposed": False,
            **(
                {"publish_capability": publish_capability}
                if platform == "facebook" and publish_capability
                else {}
            ),
            **error_details,
        }

    return {
        "mode": "manual_readonly_credential_diagnostic",
        "phase1_required_platforms": list(PLATFORM_REQUIREMENTS),
        "all_required_credentials_valid": all(
            item["status"] == "valid" for item in platforms.values()
        ),
        "publishing_attempted": False,
        "external_writes_attempted": False,
        "paid_operations_attempted": False,
        "measured_cost_usd": 0.0,
        "secret_values_exposed": False,
        "platforms": platforms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run manual read-only credential checks without exposing secrets."
        )
    )
    parser.add_argument(
        "--output", default="artifacts/platform-credential-diagnostic.json"
    )
    args = parser.parse_args()
    report = build_credential_diagnostic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Diagnóstico seguro terminado: "
        f"{sum(item['status'] == 'valid' for item in report['platforms'].values())}"
        f"/{len(report['platforms'])} plataformas válidas"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
