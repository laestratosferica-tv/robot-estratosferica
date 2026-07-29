import unittest
from unittest.mock import patch

from tools.collect_instagram_reel_insights import (
    collect_metrics,
    find_media,
    metric_value,
)


class InstagramReelInsightsTests(unittest.TestCase):
    def test_metric_value_reads_latest_value(self):
        self.assertEqual(
            metric_value({"data": [{"values": [{"value": 10}, {"value": 14}]}]}),
            14,
        )

    @patch("tools.collect_instagram_reel_insights.graph_get")
    def test_find_media_matches_normalized_permalink(self, graph_get):
        graph_get.return_value = {
            "data": [{
                "id": "123",
                "permalink": "https://www.instagram.com/reel/abc/",
            }],
        }
        with patch.dict("os.environ", {}, clear=True):
            media = find_media(
                "https://graph.facebook.com/v25.0",
                "account",
                "secret",
                "https://www.instagram.com/reel/abc",
            )
        self.assertEqual(media["id"], "123")

    @patch("tools.collect_instagram_reel_insights.graph_get")
    def test_unsupported_metric_does_not_break_collection(self, graph_get):
        def response(_base, _path, params):
            if params["metric"] == "views":
                return {"data": [{"values": [{"value": 150}]}]}
            raise RuntimeError(
                f"métrica no compatible con token {params['access_token']}"
            )

        graph_get.side_effect = response
        metrics, unavailable = collect_metrics("base", "media", "secret")
        self.assertEqual(metrics["views"], 150)
        self.assertIn("reach", unavailable)
        self.assertNotIn("secret", str(unavailable))
        self.assertIn("***", str(unavailable))


if __name__ == "__main__":
    unittest.main()
