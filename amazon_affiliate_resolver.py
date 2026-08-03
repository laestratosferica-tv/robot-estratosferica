"""Resolvedor Amazon autorizado y fail-closed para piezas comerciales.

La configuración completa vive en un único secreto JSON: ``AMAZON_PRODUCT_API_CONFIG``.
El módulo nunca imprime credenciales y no publica contenido.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from commerce_platform_routing import AFFILIATE_DISCLOSURE, build_amazon_routes


CONFIG_ENV = "AMAZON_PRODUCT_API_CONFIG"
CONFIG_FIELDS = {"access_key", "secret_key", "associate_tag", "marketplace", "host", "region"}
PAAPI_TARGET = "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems"


def _text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"[^a-z0-9]+", " ", normalized.encode("ascii", "ignore").decode().lower()).strip()


@dataclass(frozen=True)
class AmazonApiConfig:
    access_key: str
    secret_key: str
    associate_tag: str
    marketplace: str
    host: str
    region: str

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> "AmazonApiConfig":
        raw = env.get(CONFIG_ENV, "").strip()
        if not raw:
            raise ValueError("amazon_product_api_config_missing")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("amazon_product_api_config_invalid_json") from exc
        missing = sorted(name for name in CONFIG_FIELDS if not str(payload.get(name, "")).strip())
        if missing:
            raise ValueError("amazon_product_api_config_incomplete:" + ",".join(missing))
        if not str(payload["host"]).endswith("amazon.com"):
            raise ValueError("amazon_product_api_host_invalid")
        return cls(**{name: str(payload[name]).strip() for name in CONFIG_FIELDS})


def configuration_diagnostic(env: Mapping[str, str]) -> dict[str, Any]:
    """Informa únicamente presencia/ausencia; jamás devuelve valores."""
    raw = env.get(CONFIG_ENV, "").strip()
    present: dict[str, bool] = {name: False for name in sorted(CONFIG_FIELDS)}
    valid_json = False
    if raw:
        try:
            payload = json.loads(raw)
            valid_json = isinstance(payload, dict)
            if valid_json:
                present = {name: bool(str(payload.get(name, "")).strip()) for name in sorted(CONFIG_FIELDS)}
        except json.JSONDecodeError:
            pass
    return {
        "schema": "amazon_product_api_config_diagnostic_v1",
        "configuration_name": CONFIG_ENV,
        "configured": bool(raw) and valid_json and all(present.values()),
        "valid_json": valid_json,
        "fields_present": present,
        "secret_values_exposed": False,
    }


class AmazonProductSource(Protocol):
    def search_items(self, *, keywords: str, resources: list[str]) -> Mapping[str, Any]: ...


class Paapi5Client:
    """Cliente oficial PA-API 5.0 firmado con AWS SigV4."""

    def __init__(self, config: AmazonApiConfig, *, timeout: int = 15):
        self.config = config
        self.timeout = timeout

    def search_items(self, *, keywords: str, resources: list[str]) -> Mapping[str, Any]:
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        endpoint = f"https://{self.config.host}/paapi5/searchitems"
        payload = {
            "Keywords": keywords,
            "Marketplace": self.config.marketplace,
            "PartnerTag": self.config.associate_tag,
            "PartnerType": "Associates",
            "Resources": resources,
        }
        body = json.dumps(payload, separators=(",", ":"))
        headers = {
            "content-encoding": "amz-1.0",
            "content-type": "application/json; charset=utf-8",
            "host": self.config.host,
            "x-amz-target": PAAPI_TARGET,
        }
        request = AWSRequest(method="POST", url=endpoint, data=body, headers=headers)
        SigV4Auth(
            Credentials(self.config.access_key, self.config.secret_key),
            "ProductAdvertisingAPI",
            self.config.region,
        ).add_auth(request)
        response = requests.post(endpoint, data=body, headers=dict(request.headers), timeout=self.timeout)
        response.raise_for_status()
        return response.json()


SEARCH_RESOURCES = [
    "ItemInfo.Title",
    "ItemInfo.ByLineInfo",
    "ItemInfo.Features",
    "Images.Primary.Large",
    "Offers.Listings.Availability.Type",
    "Offers.Listings.Price",
]


def _affiliate_url(detail_url: str, tag: str, host: str) -> str:
    parsed = urlparse(detail_url)
    if parsed.scheme != "https" or parsed.hostname != host:
        raise ValueError("amazon_detail_url_invalid")
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["tag"] = [tag]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def resolve_product(
    request: Mapping[str, Any], config: AmazonApiConfig, source: AmazonProductSource
) -> dict[str, Any]:
    """Resuelve un único producto; cualquier ambigüedad bloquea la pieza."""
    query = str(request.get("query", "")).strip()
    expected = [_text(term) for term in request.get("expected_terms", []) if _text(term)]
    if not query or not expected:
        raise ValueError("commercial_request_query_or_expected_terms_missing")
    response = source.search_items(keywords=query, resources=SEARCH_RESOURCES)
    items = response.get("SearchResult", {}).get("Items", [])
    matches: list[dict[str, Any]] = []
    for item in items:
        title = item.get("ItemInfo", {}).get("Title", {}).get("DisplayValue", "")
        features = " ".join(item.get("ItemInfo", {}).get("Features", {}).get("DisplayValues", []))
        brand = item.get("ItemInfo", {}).get("ByLineInfo", {}).get("Brand", {}).get("DisplayValue", "")
        searchable = _text(" ".join((title, features, brand)))
        visual_url = item.get("Images", {}).get("Primary", {}).get("Large", {}).get("URL", "")
        listings = item.get("Offers", {}).get("Listings", [])
        available = any(
            _text(listing.get("Availability", {}).get("Type", "")) == "now" for listing in listings
        )
        if all(term in searchable for term in expected) and str(visual_url).startswith("https://") and available:
            matches.append({
                "asin": item.get("ASIN"),
                "title": title,
                "brand": brand,
                "detail_url": item.get("DetailPageURL"),
                "image_url": visual_url,
                "available": True,
            })
    if len(matches) != 1:
        raise ValueError(f"amazon_product_match_not_unique:{len(matches)}")
    product = matches[0]
    if not product["asin"] or not product["detail_url"]:
        raise ValueError("amazon_product_identity_incomplete")
    marketplace_host = urlparse(
        config.marketplace if "://" in config.marketplace else "https://" + config.marketplace
    ).hostname
    if not marketplace_host:
        raise ValueError("amazon_marketplace_invalid")
    affiliate_url = _affiliate_url(product["detail_url"], config.associate_tag, marketplace_host)
    routes = {name: asdict(route) for name, route in build_amazon_routes(
        affiliate_url, expected_tag=config.associate_tag, expected_host=marketplace_host
    ).items()}
    approval_verified = request.get("approval_verified") is True
    evidence = {
        "schema": "amazon_affiliate_resolution_v1",
        "request_id": request.get("request_id"),
        "publishable": approval_verified,
        "source": "amazon_paapi5",
        "marketplace": config.marketplace,
        "product": product,
        "affiliate_url": affiliate_url,
        "associate_tag_verified": True,
        "https_verified": True,
        "availability_verified": True,
        "product_match_verified": True,
        "visual_reference_verified": True,
        "approval_verified": approval_verified,
        "approval_record_path": request.get("approval_record_path"),
        "affiliate_disclosure": AFFILIATE_DISCLOSURE,
        "routes": routes,
        "secret_values_exposed": False,
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return evidence
