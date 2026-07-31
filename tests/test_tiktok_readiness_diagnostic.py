import io
import json
import urllib.error
import unittest

from tiktok_readiness_diagnostic import build_tiktok_readiness_diagnostic


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        return json.dumps(self.payload).encode("utf-8")


class TikTokReadinessDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "ENABLE_TIKTOK": "false",
            "ENABLE_TIKTOK_PUBLISH": "false",
            "TIKTOK_CLIENT_KEY": "client-key",
            "TIKTOK_CLIENT_SECRET": "client-secret",
            "TIKTOK_REDIRECT_URI": "https://example.com/oauth/tiktok/callback",
            "TIKTOK_ACCESS_TOKEN": "access-secret",
            "TIKTOK_REFRESH_TOKEN": "refresh-secret",
            "TIKTOK_OPEN_ID": "open-id",
            "TIKTOK_AUTHORIZED_SCOPES": "user.info.basic,video.upload",
            "TIKTOK_APP_REVIEW_STATUS": "approved",
            "TIKTOK_CONTENT_POSTING_API_STATUS": "approved",
        }

    def test_ready_report_uses_only_readonly_user_endpoint(self):
        requests = []

        def opener(request, timeout):
            self.assertEqual(timeout, 15)
            requests.append(request)
            return _Response(
                {
                    "data": {
                        "user": {
                            "open_id": "open-id",
                            "display_name": "La Estratosférica",
                        }
                    },
                    "error": {"code": "ok", "message": ""},
                }
            )

        report = build_tiktok_readiness_diagnostic(
            self.environment,
            opener=opener,
        )

        self.assertTrue(report["ready_for_private_test"])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "GET")
        self.assertIn("/v2/user/info/", requests[0].full_url)
        self.assertNotIn("/post/publish/", requests[0].full_url)
        self.assertFalse(report["publishing_attempted"])
        self.assertFalse(report["upload_attempted"])
        self.assertFalse(report["external_writes_attempted"])
        rendered = json.dumps(report)
        for secret in ("client-secret", "access-secret", "refresh-secret"):
            self.assertNotIn(secret, rendered)

    def test_missing_configuration_does_not_call_network(self):
        def opener(_request, _timeout):
            raise AssertionError("network must not be called")

        report = build_tiktok_readiness_diagnostic({}, opener=opener)

        self.assertFalse(report["ready_for_private_test"])
        self.assertIn("oauth_configuration_incomplete", report["blockers"])
        self.assertIn("tiktok_approvals_not_confirmed", report["blockers"])
        self.assertEqual(
            report["live_readonly_identity_check"]["status"],
            "not_run",
        )

    def test_enabled_publish_flag_is_always_a_blocker(self):
        environment = dict(self.environment)
        environment["ENABLE_TIKTOK_PUBLISH"] = "true"

        report = build_tiktok_readiness_diagnostic(
            environment,
            opener=lambda _request, timeout: _Response(
                {
                    "data": {"user": {"open_id": "open-id"}},
                    "error": {"code": "ok"},
                }
            ),
        )

        self.assertFalse(report["ready_for_private_test"])
        self.assertIn("publishing_flags_must_remain_false", report["blockers"])
        self.assertFalse(report["publishing_attempted"])

    def test_scope_and_identity_mismatch_are_reported(self):
        environment = dict(self.environment)
        environment["TIKTOK_AUTHORIZED_SCOPES"] = "user.info.basic"

        report = build_tiktok_readiness_diagnostic(
            environment,
            opener=lambda _request, timeout: _Response(
                {
                    "data": {"user": {"open_id": "different-open-id"}},
                    "error": {"code": "ok"},
                }
            ),
        )

        self.assertIn("content_posting_scope_missing", report["blockers"])
        self.assertIn("configured_open_id_mismatch", report["blockers"])

    def test_provider_error_is_redacted(self):
        def opener(request, timeout):
            payload = {
                "error": {
                    "code": "access_token_invalid",
                    "message": "access_token=access-secret is invalid",
                    "log_id": "provider-log",
                }
            }
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "access-secret",
                {},
                io.BytesIO(json.dumps(payload).encode("utf-8")),
            )

        report = build_tiktok_readiness_diagnostic(
            self.environment,
            opener=opener,
        )

        self.assertEqual(
            report["live_readonly_identity_check"]["status"],
            "provider_rejected",
        )
        rendered = json.dumps(report)
        self.assertNotIn("access-secret", rendered)
        self.assertNotIn("client-secret", rendered)
        self.assertNotIn("refresh-secret", rendered)


if __name__ == "__main__":
    unittest.main()
