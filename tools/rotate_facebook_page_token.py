#!/usr/bin/env python3
"""Prepare and execute an auditable Facebook Page Token rotation."""

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import requests


GRAPH_BASE = "https://graph.facebook.com"
SCOPES = ("pages_show_list", "pages_read_engagement", "pages_manage_posts")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(env: Mapping[str, str], *names: str) -> None:
    missing = [name for name in names if not env.get(name, "").strip()]
    if missing:
        raise RuntimeError("missing_protected_configuration:" + ",".join(missing))


def safe_error(response: requests.Response) -> str:
    try:
        error = response.json().get("error", {})
    except ValueError:
        error = {}
    return "meta_request_failed:" + ":".join(
        str(error.get(key, "unknown")) for key in ("type", "code", "error_subcode")
    )


def graph_request(
    method: str, path: str, *, version: str, token: str | None = None,
    params: dict[str, str] | None = None, data: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.request(
        method, f"{GRAPH_BASE}/{version}/{path.lstrip('/')}", headers=headers,
        params=params, data=data, timeout=45,
    )
    if not response.ok:
        raise RuntimeError(safe_error(response))
    payload = response.json()
    if not isinstance(payload, dict) or "error" in payload:
        raise RuntimeError("meta_response_invalid")
    return payload


def authorization_url(env: Mapping[str, str]) -> str:
    require(env, "FB_APP_ID", "FB_OAUTH_REDIRECT_URI")
    query = urlencode({
        "client_id": env["FB_APP_ID"],
        "redirect_uri": env["FB_OAUTH_REDIRECT_URI"],
        "response_type": "code",
        "scope": ",".join(SCOPES),
    })
    return f"https://www.facebook.com/{env.get('GRAPH_VERSION', 'v25.0')}/dialog/oauth?{query}"


def exchange_code(env: Mapping[str, str]) -> str:
    require(env, "FB_APP_ID", "FB_APP_SECRET", "FB_OAUTH_REDIRECT_URI", "FB_AUTH_CODE")
    payload = graph_request(
        "POST", "oauth/access_token", version=env.get("GRAPH_VERSION", "v25.0"),
        data={"client_id": env["FB_APP_ID"], "client_secret": env["FB_APP_SECRET"],
              "redirect_uri": env["FB_OAUTH_REDIRECT_URI"], "code": env["FB_AUTH_CODE"]},
    )
    token = str(payload.get("access_token", ""))
    if not token:
        raise RuntimeError("facebook_oauth_code_exchange_missing_token")
    return token


def exchange_long_lived_token(short_token: str, env: Mapping[str, str]) -> str:
    payload = graph_request(
        "GET", "oauth/access_token", version=env.get("GRAPH_VERSION", "v25.0"),
        params={"grant_type": "fb_exchange_token", "client_id": env["FB_APP_ID"],
                "client_secret": env["FB_APP_SECRET"], "fb_exchange_token": short_token},
    )
    token = str(payload.get("access_token", ""))
    if not token:
        raise RuntimeError("facebook_long_lived_exchange_missing_token")
    return token


def page_token(user_token: str, env: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    require(env, "FB_PAGE_ID")
    payload = graph_request(
        "GET", "me/accounts", version=env.get("GRAPH_VERSION", "v25.0"), token=user_token,
        params={"fields": "id,name,access_token"},
    )
    for page in payload.get("data", []):
        if str(page.get("id")) == env["FB_PAGE_ID"] and page.get("access_token"):
            return str(page["access_token"]), page
    raise RuntimeError("authorized_user_does_not_manage_expected_page")


def verify_page(token: str, env: Mapping[str, str]) -> dict[str, Any]:
    profile = graph_request(
        "GET", env["FB_PAGE_ID"], version=env.get("GRAPH_VERSION", "v25.0"), token=token,
        params={"fields": "id,name,link"},
    )
    if str(profile.get("id")) != env["FB_PAGE_ID"]:
        raise RuntimeError("facebook_page_identity_mismatch")
    return {"id": profile.get("id"), "name": profile.get("name"), "link": profile.get("link")}


def rotate(env: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    short = exchange_code(env)
    long_lived = exchange_long_lived_token(short, env)
    token, page = page_token(long_lived, env)
    profile = verify_page(token, env)
    return token, {"page": profile, "page_selected_name": page.get("name"), "rotated_at": utc_now()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "rotate"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-output")
    args = parser.parse_args()
    if args.mode == "prepare":
        report = {"mode": "prepare", "authorization_url": authorization_url(os.environ),
                  "scopes": list(SCOPES), "external_writes_attempted": False,
                  "secret_values_exposed": False, "prepared_at": utc_now()}
    else:
        if not args.token_output:
            raise RuntimeError("token_output_required_for_rotation")
        token, metadata = rotate(os.environ)
        token_path = Path(args.token_output)
        token_path.write_text(token, encoding="utf-8")
        token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        report = {"mode": "rotate", "status": "page_token_verified", **metadata,
                  "secret_values_exposed": False}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Facebook OAuth preparado." if args.mode == "prepare" else "Token de página verificado y listo para guardar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
