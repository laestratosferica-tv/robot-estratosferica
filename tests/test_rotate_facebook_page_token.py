import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from tools.rotate_facebook_page_token import authorization_url, page_token


class FacebookTokenRotationTests(unittest.TestCase):
    def test_prepare_uses_only_required_page_scopes(self):
        url = authorization_url({"FB_APP_ID": "app-id", "FB_OAUTH_REDIRECT_URI": "https://example.com/callback"})
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["client_id"], ["app-id"])
        self.assertEqual(query["redirect_uri"], ["https://example.com/callback"])
        self.assertEqual(query["scope"], ["pages_show_list,pages_read_engagement,pages_manage_posts"])

    def test_page_token_rejects_another_page(self):
        with patch(
            "tools.rotate_facebook_page_token.graph_request",
            return_value={"data": [{"id": "another", "access_token": "token"}]},
        ):
            with self.assertRaisesRegex(RuntimeError, "expected_page"):
                page_token("user-token", {"FB_PAGE_ID": "expected"})


if __name__ == "__main__":
    unittest.main()
