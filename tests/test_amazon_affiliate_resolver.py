import json
import unittest

from amazon_affiliate_resolver import (
    AmazonApiConfig,
    configuration_diagnostic,
    resolve_product,
)


class FakeSource:
    def __init__(self, items):
        self.items = items

    def search_items(self, *, keywords, resources):
        return {"SearchResult": {"Items": self.items}}


def item(*, asin="B012345678", available="Now", title="Mini wireless microphone USB C"):
    return {
        "ASIN": asin,
        "DetailPageURL": f"https://www.amazon.com/dp/{asin}?ref=paapi",
        "ItemInfo": {
            "Title": {"DisplayValue": title},
            "ByLineInfo": {"Brand": {"DisplayValue": "Example Brand"}},
            "Features": {"DisplayValues": ["For creators"]},
        },
        "Images": {"Primary": {"Large": {"URL": "https://images.amazon.com/product.jpg"}}},
        "Offers": {"Listings": [{"Availability": {"Type": available}}]},
    }


class AmazonAffiliateResolverTests(unittest.TestCase):
    def setUp(self):
        self.config = AmazonApiConfig(
            access_key="key",
            secret_key="secret",
            associate_tag="estrato-test-20",
            marketplace="www.amazon.com",
            host="webservices.amazon.com",
            region="us-east-1",
        )
        self.request = {
            "request_id": "mini-mic-test",
            "query": "mini microphone",
            "expected_terms": ["wireless", "microphone", "usb c"],
            "approval_verified": True,
            "approval_record_path": "artifacts/approval-records/test.json",
        }

    def test_diagnostic_never_exposes_values(self):
        payload = {name: name + "-private" for name in (
            "access_key", "secret_key", "associate_tag", "marketplace", "host", "region"
        )}
        report = configuration_diagnostic({"AMAZON_PRODUCT_API_CONFIG": json.dumps(payload)})
        self.assertTrue(report["configured"])
        self.assertFalse(report["secret_values_exposed"])
        self.assertNotIn("private", json.dumps(report))

    def test_resolves_unique_available_authoritative_product(self):
        evidence = resolve_product(self.request, self.config, FakeSource([item()]))
        self.assertTrue(evidence["publishable"])
        self.assertIn("tag=estrato-test-20", evidence["affiliate_url"])
        self.assertEqual(len(evidence["routes"]), 5)
        self.assertTrue(evidence["visual_reference_verified"])

    def test_fails_closed_when_unavailable(self):
        with self.assertRaisesRegex(ValueError, "not_unique:0"):
            resolve_product(self.request, self.config, FakeSource([item(available="Future")]))

    def test_fails_closed_when_match_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "not_unique:2"):
            resolve_product(self.request, self.config, FakeSource([item(), item(asin="B087654321")]))

    def test_missing_bundle_is_single_explicit_blocker(self):
        with self.assertRaisesRegex(ValueError, "amazon_product_api_config_missing"):
            AmazonApiConfig.from_environment({})


if __name__ == "__main__":
    unittest.main()
