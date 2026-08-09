from __future__ import annotations

import unittest

from tools.assess_amazon_influencer_readiness import assess_sustained_readiness


def snapshot(
    followers: int,
    posts: int,
    engagement: float,
    growth: float,
    quality: str = "complete",
) -> dict:
    return {
        "schema_version": "instagram_account_growth_v1",
        "internal_readiness": {
            "data_quality": quality,
            "targets": {
                "followers": 2000,
                "weekly_posts": 3,
                "engagement_rate": 0.03,
            },
            "observed": {
                "followers": followers,
                "weekly_posts": posts,
                "engagement_rate": engagement,
                "weekly_follower_growth_rate": growth,
            },
        },
    }


class AmazonInfluencerReadinessTests(unittest.TestCase):
    def test_requires_four_weeks(self) -> None:
        report = assess_sustained_readiness([snapshot(2100, 4, 0.04, 0.02)])
        self.assertFalse(report["ready_for_human_reapplication_review"])
        self.assertFalse(report["checks"]["four_week_history"])

    def test_all_sustained_internal_gates_can_pass(self) -> None:
        report = assess_sustained_readiness(
            [
                snapshot(2010, 4, 0.04, 0.01),
                snapshot(2040, 3, 0.035, 0.015),
                snapshot(2060, 5, 0.032, 0.01),
                snapshot(2080, 3, 0.028, 0.01),
            ]
        )
        self.assertTrue(report["ready_for_human_reapplication_review"])
        self.assertFalse(report["amazon_approval_guaranteed"])
        self.assertFalse(report["automatic_reapplication_enabled"])

    def test_one_weak_gate_blocks_review(self) -> None:
        report = assess_sustained_readiness(
            [
                snapshot(2100, 2, 0.04, 0.01),
                snapshot(2120, 3, 0.04, 0.01),
                snapshot(2140, 3, 0.04, 0.01),
                snapshot(2160, 3, 0.04, 0.01),
            ]
        )
        self.assertFalse(report["checks"]["posting_consistency"])
        self.assertFalse(report["ready_for_human_reapplication_review"])


if __name__ == "__main__":
    unittest.main()
