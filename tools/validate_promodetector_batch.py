#!/usr/bin/env python3
"""Fail-closed validator for a PromoDetector review batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
LANDING = ROOT / "artifacts" / "review" / "promodetector-v1"
DEFAULT_MANIFEST = LANDING / "promodetector-batch-01.json"
DISCLOSURE = "Como Afiliado de Amazon"


def validate(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    errors: list[str] = []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    summary = data.get("summary", {})

    if data.get("publication_allowed") is not False:
        errors.append("publication_allowed debe permanecer en false durante revisión")
    if summary.get("ready") != len(items):
        errors.append("summary.ready no coincide con el número de fichas")

    for item in items:
        slug = item.get("slug", "sin-slug")
        page = LANDING / item.get("page", "")
        if not page.is_file():
            errors.append(f"{slug}: falta la ficha local")
            continue
        html = page.read_text(encoding="utf-8")
        if item.get("status") != "ready":
            errors.append(f"{slug}: un elemento no listo entró al bloque público")

        if item.get("kind") == "commercial":
            evidence = (manifest_path.parent / item.get("evidence", "")).resolve()
            if not evidence.is_file():
                errors.append(f"{slug}: falta evidencia comercial")
                continue
            proof = json.loads(evidence.read_text(encoding="utf-8"))
            affiliate_url = proof.get("affiliate_url", "")
            parsed = urlparse(affiliate_url)
            if proof.get("publishable") is not True:
                errors.append(f"{slug}: evidencia no publicable")
            if parsed.scheme != "https" or not parsed.netloc.endswith("amazon.com"):
                errors.append(f"{slug}: destino Amazon inválido")
            if "tag=" not in parsed.query:
                errors.append(f"{slug}: falta associate tag")
            if affiliate_url not in html:
                errors.append(f"{slug}: la ficha no usa el enlace aprobado")
            if DISCLOSURE not in html:
                errors.append(f"{slug}: falta divulgación afiliada")
        elif item.get("kind") == "editorial":
            source = item.get("source", "")
            if urlparse(source).scheme != "https":
                errors.append(f"{slug}: fuente editorial no segura")
            if DISCLOSURE in html:
                errors.append(f"{slug}: ficha editorial marcada incorrectamente como afiliada")
        else:
            errors.append(f"{slug}: tipo desconocido")

    return errors


def main() -> int:
    manifest = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_MANIFEST
    errors = validate(manifest)
    if errors:
        print("PROMODETECTOR_BATCH_BLOCKED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PROMODETECTOR_BATCH_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
