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


TIKTOK_USER_INFO_API = "https://open.tiktokapis.com/v2/user/info/"

OAUTH_REQUIREMENTS = (
    "TIKTOK_CLIENT_KEY",
    "TIKTOK_CLIENT_SECRET",
    "TIKTOK_REDIRECT_URI",
    "TIKTOK_ACCESS_TOKEN",
    "TIKTOK_REFRESH_TOKEN",
    "TIKTOK_OPEN_ID",
    "TIKTOK_AUTHORIZED_SCOPES",
)

SECRET_NAMES = (
    "TIKTOK_CLIENT_SECRET",
    "TIKTOK_ACCESS_TOKEN",
    "TIKTOK_REFRESH_TOKEN",
)

PUBLISH_FLAGS = (
    "ENABLE_TIKTOK",
    "ENABLE_TIKTOK_PUBLISH",
)


def _env_bool(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_scopes(raw: str) -> set[str]:
    return {
        value.strip()
        for value in re.split(r"[\s,]+", raw)
        if value.strip()
    }


def _redact(value: Any, secrets: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    safe = value
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        safe = safe.replace(secret, "***")
    safe = re.sub(
        r"(?i)(access[_ -]?token|refresh[_ -]?token|client[_ -]?secret)"
        r"([\"'=:\s]+)[^\s,\"'}]+",
        r"\1\2***",
        safe,
    )
    return safe[:500]


def _safe_error(error: Exception, secrets: list[str]) -> dict[str, Any]:
    if isinstance(error, urllib.error.HTTPError):
        details: dict[str, Any] = {"http_status": error.code}
        try:
            payload = json.loads(error.read(65536).decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            payload = {}
        finally:
            error.close()
        provider_error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(provider_error, dict):
            for source, target in (
                ("code", "provider_error_code"),
                ("message", "provider_message"),
                ("log_id", "provider_log_id"),
            ):
                value = provider_error.get(source)
                if value:
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


def _readonly_user_check(
    *,
    access_token: str,
    expected_open_id: str,
    opener: Any,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({"fields": "open_id,display_name"})
    request = urllib.request.Request(
        f"{TIKTOK_USER_INFO_API}?{query}",
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    with opener(request, timeout=15) as response:
        payload = json.loads(response.read(65536).decode("utf-8"))

    error = payload.get("error", {})
    if error and error.get("code") not in {None, "", "ok"}:
        raise RuntimeError("TikTok returned an application-level error")

    user = payload.get("data", {}).get("user", {})
    returned_open_id = str(user.get("open_id", "")).strip()
    return {
        "status": "valid" if returned_open_id else "invalid_response",
        "open_id_matches_configured": bool(
            returned_open_id and returned_open_id == expected_open_id
        ),
        "profile_display_name_present": bool(user.get("display_name")),
    }


def build_tiktok_readiness_diagnostic(
    environment: Mapping[str, str] | None = None,
    *,
    opener: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    values = {name: environment.get(name, "").strip() for name in OAUTH_REQUIREMENTS}
    secrets = [environment.get(name, "").strip() for name in SECRET_NAMES]
    missing = [name for name, value in values.items() if not value]
    scopes = _parse_scopes(values["TIKTOK_AUTHORIZED_SCOPES"])
    flags = {name: _env_bool(environment, name) for name in PUBLISH_FLAGS}
    flags_safely_disabled = not any(flags.values())

    app_review_status = environment.get("TIKTOK_APP_REVIEW_STATUS", "").strip().lower()
    posting_api_status = (
        environment.get("TIKTOK_CONTENT_POSTING_API_STATUS", "").strip().lower()
    )
    approvals_confirmed = (
        app_review_status == "approved" and posting_api_status == "approved"
    )

    has_read_scope = "user.info.basic" in scopes
    has_posting_scope = bool({"video.upload", "video.publish"} & scopes)
    live_check: dict[str, Any] = {
        "status": "not_run",
        "reason": "missing_access_token_or_user_info_basic_scope",
    }

    if values["TIKTOK_ACCESS_TOKEN"] and has_read_scope:
        try:
            live_check = _readonly_user_check(
                access_token=values["TIKTOK_ACCESS_TOKEN"],
                expected_open_id=values["TIKTOK_OPEN_ID"],
                opener=opener,
            )
        except Exception as error:
            live_check = {
                "status": _safe_status(error),
                **_safe_error(error, secrets),
            }

    blockers: list[str] = []
    if not flags_safely_disabled:
        blockers.append("publishing_flags_must_remain_false")
    if missing:
        blockers.append("oauth_configuration_incomplete")
    if not approvals_confirmed:
        blockers.append("tiktok_approvals_not_confirmed")
    if not has_read_scope:
        blockers.append("user_info_basic_scope_missing")
    if not has_posting_scope:
        blockers.append("content_posting_scope_missing")
    if live_check.get("status") != "valid":
        blockers.append("readonly_identity_check_not_valid")
    elif not live_check.get("open_id_matches_configured"):
        blockers.append("configured_open_id_mismatch")

    ready_for_private_test = not blockers

    return {
        "mode": "tiktok_readonly_readiness_diagnostic",
        "publishing_attempted": False,
        "upload_attempted": False,
        "external_writes_attempted": False,
        "paid_operations_attempted": False,
        "secret_values_exposed": False,
        "publishing_flags": {
            name: {"enabled": enabled, "required_state": False}
            for name, enabled in flags.items()
        },
        "publishing_flags_safely_disabled": flags_safely_disabled,
        "oauth_configuration": {
            "configured_count": len(values) - len(missing),
            "required_count": len(values),
            "missing_variable_names": missing,
        },
        "declared_scopes": {
            "user_info_basic": has_read_scope,
            "video_upload": "video.upload" in scopes,
            "video_publish": "video.publish" in scopes,
        },
        "approval_status": {
            "app_review": app_review_status or "unknown",
            "content_posting_api": posting_api_status or "unknown",
            "confirmed": approvals_confirmed,
        },
        "live_readonly_identity_check": live_check,
        "ready_for_private_test": ready_for_private_test,
        "blockers": blockers,
        "next_action": (
            "prepare_single_private_test_plan"
            if ready_for_private_test
            else "resolve_blockers_without_enabling_publishing"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose TikTok OAuth and approval readiness without uploading or "
            "publishing content."
        )
    )
    parser.add_argument(
        "--output",
        default="artifacts/tiktok-readiness-diagnostic.json",
    )
    args = parser.parse_args()
    report = build_tiktok_readiness_diagnostic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Diagnóstico TikTok seguro terminado: "
        f"{len(report['blockers'])} bloqueadores; "
        "publicación y carga no ejecutadas"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
