import json
import time
import unittest
import urllib.parse

from tiktok_creator_portal import (
    INBOX_INIT_URL,
    Settings,
    TikTokClient,
    _signed_state,
    _valid_state,
)


class _Response:
    def __init__(self, payload=b"", status=200):
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        return self.payload


def settings(**overrides):
    values = {
        "client_key": "client-key",
        "client_secret": "client-secret",
        "redirect_uri": "https://example.com/oauth/tiktok/callback",
        "session_secret": "session-secret-that-is-long",
        "public_base_url": "https://example.com",
        "draft_transfer_enabled": False,
        "sandbox_review_mode": False,
        "app_review_status": "rejected",
    }
    values.update(overrides)
    return Settings(**values)


class TikTokCreatorPortalTests(unittest.TestCase):
    def test_safe_defaults_block_external_transfer(self):
        configured = Settings.from_environment(
            {
                "TIKTOK_CLIENT_KEY": "key",
                "TIKTOK_CLIENT_SECRET": "secret",
                "TIKTOK_REDIRECT_URI": "https://example.com/callback",
                "TIKTOK_SESSION_SECRET": "session",
            }
        )
        self.assertTrue(configured.oauth_configured)
        self.assertFalse(configured.draft_transfer_enabled)
        self.assertFalse(configured.sandbox_review_mode)
        self.assertFalse(configured.transfer_allowed)

    def test_transfer_requires_explicit_flag_and_sandbox_or_approval(self):
        self.assertFalse(settings(draft_transfer_enabled=True).transfer_allowed)
        self.assertTrue(
            settings(draft_transfer_enabled=True, sandbox_review_mode=True).transfer_allowed
        )
        self.assertTrue(
            settings(draft_transfer_enabled=True, app_review_status="approved").transfer_allowed
        )

    def test_oauth_url_requests_only_required_scopes(self):
        client = TikTokClient(settings())
        url = client.authorization_url("signed-state")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query["scope"], ["user.info.basic,video.upload"])
        self.assertNotIn("video.publish", query["scope"][0])
        self.assertEqual(query["state"], ["signed-state"])

    def test_signed_state_is_bound_to_session_and_expires(self):
        state = _signed_state("session-a", settings())
        self.assertTrue(_valid_state(state, "session-a", settings()))
        self.assertFalse(_valid_state(state, "session-b", settings()))
        parts = state.rsplit(".", 3)
        parts[1] = str(int(time.time()) - 601)
        body = ".".join(parts[:3])
        import hashlib, hmac

        parts[3] = hmac.new(b"session-secret-that-is-long", body.encode(), hashlib.sha256).hexdigest()
        self.assertFalse(_valid_state(".".join(parts), "session-a", settings()))

    def test_send_to_inbox_uses_draft_endpoint_and_one_upload(self):
        requests = []

        def opener(request, timeout):
            requests.append((request, timeout))
            if request.full_url == INBOX_INIT_URL:
                return _Response(
                    {
                        "data": {"upload_url": "https://upload.example/video", "publish_id": "draft-1"},
                        "error": {"code": "ok"},
                    }
                )
            return _Response(b"")

        receipt = TikTokClient(settings(), opener=opener).send_to_inbox(
            "access-token", b"video-bytes", "video/mp4"
        )
        self.assertEqual(receipt, "draft-1")
        self.assertEqual([request.method for request, _ in requests], ["POST", "PUT"])
        self.assertIn("/post/publish/inbox/video/init/", requests[0][0].full_url)
        self.assertNotIn("direct_post", requests[0][0].full_url)
        self.assertEqual(requests[1][0].headers["Content-range"], "bytes 0-10/11")


if __name__ == "__main__":
    unittest.main()

