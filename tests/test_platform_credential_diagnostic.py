import io
import json
import urllib.error
import unittest

from platform_credential_diagnostic import build_credential_diagnostic


class _Response:
    def __init__(self, body=b"{"):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        return self.body


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
            if request.full_url.endswith("/me/permissions"):
                return _Response(
                    json.dumps(
                        {
                            "data": [
                                {
                                    "permission": "pages_manage_posts",
                                    "status": "granted",
                                }
                            ]
                        }
                    ).encode()
                )
            if request.full_url.endswith("/me?fields=id"):
                return _Response(
                    json.dumps({"id": "facebook-id"}).encode()
                )
            return _Response()

        report = build_credential_diagnostic(self.environment, opener=opener)

        self.assertTrue(report["all_required_credentials_valid"])
        self.assertEqual(len(requests), 6)
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
        facebook = report["platforms"]["facebook"]["publish_capability"]
        self.assertTrue(facebook["token_subject_matches_page"])
        self.assertTrue(facebook["pages_manage_posts_confirmed"])
        self.assertTrue(facebook["reels_publish_permission_ready"])
        self.assertTrue(
            facebook["page_content_task_requires_business_suite_verification"]
        )

    def test_missing_and_incomplete_credentials_do_not_call_network(self):
        def opener(_request, _timeout):
            raise AssertionError("network must not be called")

        report = build_credential_diagnostic(
            {"THREADS_USER_ACCESS_TOKEN": "secret"}, opener=opener
        )

        self.assertEqual(report["platforms"]["threads"]["status"], "incomplete")
        self.assertEqual(report["platforms"]["instagram"]["status"], "missing")
        self.assertFalse(report["all_required_credentials_valid"])

    def test_provider_failure_keeps_safe_cause_and_redacts_secrets(self):
        def opener(request, timeout):
            body = {
                "error": {
                    "message": (
                        "Invalid OAuth access token instagram-secret "
                        "access_token=another-sensitive-value"
                    ),
                    "type": "OAuthException",
                    "code": 190,
                    "error_subcode": 463,
                }
            }
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "unauthorized-secret-detail",
                {},
                io.BytesIO(json.dumps(body).encode()),
            )

        report = build_credential_diagnostic(self.environment, opener=opener)

        self.assertTrue(
            all(
                item["status"] == "provider_rejected"
                for item in report["platforms"].values()
            )
        )
        instagram = report["platforms"]["instagram"]
        self.assertEqual(instagram["http_status"], 401)
        self.assertEqual(instagram["provider_error_code"], 190)
        self.assertEqual(instagram["provider_error_subcode"], 463)
        self.assertEqual(instagram["provider_error_type"], "OAuthException")
        rendered = json.dumps(report)
        self.assertNotIn("instagram-secret", rendered)
        self.assertNotIn("another-sensitive-value", rendered)
        self.assertNotIn("unauthorized-secret-detail", rendered)

    def test_non_json_failure_reports_http_status_only(self):
        def opener(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "forbidden",
                {},
                io.BytesIO(b"not-json"),
            )

        report = build_credential_diagnostic(self.environment, opener=opener)

        self.assertEqual(
            report["platforms"]["facebook"],
            {
                "status": "provider_rejected",
                "configured_count": 2,
                "required_count": 2,
                "live_readonly_check_performed": True,
                "secret_values_exposed": False,
                "http_status": 403,
            },
        )


if __name__ == "__main__":
    unittest.main()
