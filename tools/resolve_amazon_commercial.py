#!/usr/bin/env python3
"""Resuelve solicitudes comerciales Amazon sin publicar."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amazon_affiliate_resolver import (
    AmazonApiConfig,
    Paapi5Client,
    configuration_diagnostic,
    resolve_product,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-id")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.diagnostic:
        result = configuration_diagnostic(os.environ)
    else:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if isinstance(request.get("items"), list):
            candidates = request["items"]
            if args.request_id:
                candidates = [item for item in candidates if item.get("request_id") == args.request_id]
            if len(candidates) != 1:
                raise SystemExit("Debe indicar --request-id para seleccionar una solicitud única")
            request = candidates[0]
        try:
            config = AmazonApiConfig.from_environment(os.environ)
            result = resolve_product(request, config, Paapi5Client(config))
        except Exception as exc:
            result = {
                "schema": "amazon_affiliate_resolution_v1",
                "request_id": request.get("request_id"),
                "publishable": False,
                "blocker": str(exc).split(":", 1)[0],
                "secret_values_exposed": False,
            }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESUELTA" if result.get("publishable") else "BLOQUEADA")
    return 0 if result.get("publishable") or args.diagnostic else 2


if __name__ == "__main__":
    raise SystemExit(main())
