from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AmazonInfluencerGrowthStrategyTests(unittest.TestCase):
    def test_strategy_prioritizes_quality_without_disabling_commerce(self) -> None:
        payload = json.loads(
            (ROOT / "config/amazon_influencer_growth_strategy_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(
            payload["volume_policy"]["increase_volume_when_engagement_is_weak"]
        )
        self.assertTrue(
            payload["commercial_continuity"]["amazon_affiliates_enabled"]
        )
        self.assertTrue(payload["commercial_continuity"]["promodetector_enabled"])
        self.assertFalse(
            payload["decision_rules"]["automatic_reapplication_enabled"]
        )

    def test_experiment_mix_is_bounded(self) -> None:
        payload = json.loads(
            (ROOT / "config/amazon_influencer_growth_strategy_v1.json").read_text(
                encoding="utf-8"
            )
        )
        mix = payload["weekly_experiment_mix"]
        self.assertEqual(mix["commercial_posts_maximum"], 1)
        self.assertGreaterEqual(
            mix["discovery_reels"] + mix["useful_carousels"], 3
        )


if __name__ == "__main__":
    unittest.main()
