from pathlib import Path
import unittest
from unittest.mock import patch

from tools.collect_instagram_account_growth import (
    build_internal_readiness,
    collect_weekly_insights,
    metric_total,
)


class InstagramAccountGrowthTests(unittest.TestCase):
    def test_workflow_is_manual_read_only_and_cannot_publish(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (
            root / ".github" / "workflows" / "instagram-account-growth-readonly.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("python tools/collect_instagram_account_growth.py", workflow)
        self.assertNotIn("media_publish", workflow)
        self.assertNotIn("PRODUCTION_ARMED", workflow)

    def test_metric_total_adds_daily_values(self):
        self.assertEqual(
            metric_total({"data": [{"values": [{"value": 10}, {"value": 14}]}]}),
            24,
        )

    @patch("tools.collect_instagram_account_growth.graph_get")
    def test_collection_is_fail_soft_and_redacts_token(self, graph_get):
        def response(_base, _path, params):
            if params["metric"] == "reach":
                return {"data": [{"values": [{"value": 100}]}]}
            raise RuntimeError(f"error using {params['access_token']}")

        graph_get.side_effect = response
        insights, unavailable = collect_weekly_insights(
            "base", "account", "secret", 1, 2
        )
        self.assertEqual(insights["reach"], 100)
        self.assertNotIn("secret", str(unavailable))
        self.assertIn("***", str(unavailable))

    @patch.dict(
        "os.environ",
        {
            "INTERNAL_FOLLOWER_TARGET": "2000",
            "INTERNAL_WEEKLY_POST_TARGET": "3",
            "INTERNAL_ENGAGEMENT_TARGET": "0.03",
            "INTERNAL_GROWTH_TARGET": "0.01",
        },
        clear=False,
    )
    def test_readiness_requires_real_baseline_and_does_not_claim_amazon_score(self):
        current = {"followers_count": 2000, "media_count": 20}
        previous = {"account": {"followers_count": 1980, "media_count": 17}}
        report = build_internal_readiness(
            current,
            {"reach": 1000, "total_interactions": 30},
            previous,
        )
        self.assertEqual(report["score"], 100.0)
        self.assertEqual(report["data_quality"], "complete")
        self.assertFalse(report["amazon_threshold_claimed"])
        self.assertFalse(report["automatic_reapplication_enabled"])
        self.assertEqual(report["recommendation"], "review_reapplication_manually")

    def test_first_snapshot_stays_partial(self):
        report = build_internal_readiness(
            {"followers_count": 802, "media_count": 10},
            {"reach": 500, "total_interactions": 20},
        )
        self.assertEqual(report["data_quality"], "partial")
        self.assertEqual(report["recommendation"], "continue_growth_experiments")


if __name__ == "__main__":
    unittest.main()
