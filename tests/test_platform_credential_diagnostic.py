import io
import json
import urllib.error
import unittest

from platform_credential_diagnostic import build_credential_diagnostic


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        return b"{"


class PlatformCredentialDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "THREADS_USER_ACCESS_TOKEN": "threads-secret",
            "THREADS_USER_ID": "threads-id",
            "IG_ACCESS_TOKEN": "instagram-secret",
            "IG_USER_ID": "instagram-id",
            "FB_PAGE_ACCESS_TOKEN": "facebook-secret",
            "FB_PAGE_ID": "facebook-id",
            "YOUTUBE_CLIENT_ID": "youtube-client",
            "YOUTUBE_CLIENT_SECRET": "youtube-secret",
            "YOUTUBE_REFRESH_TOKEN": "youtube-refresh",
        }

    def test_validates_four_required_platforms_without_exposing_secrets(self):
        requests = []

        def opener(request, timeout):
            self.assertEqual(timeout, 15)
            requests.append(request)
            return _Response()

        report = build_credential_diagnostic(
            self.environment, opener=opener
        )

        self.assertTrue(report["all_required_credentials_valid"])
        self.assertEqual(len(requests), 4)
        for request in requests[:3]:
            self.assertNotIn("secret", request.full_url)
            self.assertTrue(
                request.get_header("Authorization").startswith("Bearer ")
            )
        self.assertTrue(
            all(
                item["status"] == "valid"
                for item in report["platforms"].values()
            )
        )
        rendered = json.dumps(report)
        for secret in self.environment.values():
            self.assertNotIn(secret, rendered)
        self.assertFalse(report["publishing_attempted"])
        self.assertFalse(report["external_writes_attempted"])
        self.assertEqual(report["measured_cost_usd"], 0.0)

    def test_missing_and_incomplete_credentials_do_not_call_network(self):
        def opener(_request, _timeout):
            raise AssertionError("network must not be called")

        report = build_credential_diagnostic(
            {"THREADS_USER_ACCESS_TOKEN": "secret"}, opener=opener
        )

        self.assertEqual(report["platforms"]["threads"]["status"], "incomplete")
        self.assertEqual(report["platforms"]["instagram"]["status"], "missing")
        self.assertFalse(report["all_required_credentials_valid"])

    def test_authentication_failure_is_redacted_and_classified(self):
        def opener(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "unauthorized-secret-detail",
                {},
                io.BytesIO(b"private response"),
            )

        report = build_credential_diagnostic(
            self.environment, opener=opener
        )

        self.assertTrue(
            all(
                item["status"] == "invalid_or_expired"
                for item in report["platforms"].values()
            )
        )
        rendered = json.dumps(report)
        self.assertNotIn("unauthorized-secret-detail", rendered)
        self.assertNotIn("private response", rendered)


if __name__ == "__main__":
    unittest.main()
