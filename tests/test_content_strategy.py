import random
import unittest

from content_strategy import classify_item, is_editorially_safe, rank_articles_by_strategy
from editorial_planner import build_editorial_plan


class ContentStrategyTests(unittest.TestCase):
    def test_classifies_core_pillars(self):
        self.assertEqual(classify_item({"title": "VALORANT final", "link": "https://dotesports.com/x"}), "gaming")
        self.assertEqual(classify_item({"title": "New AI creator tool", "link": "https://blog.google/x"}), "technology")
        self.assertEqual(classify_item({"title": "Campaign strategy", "link": "https://blog.hubspot.com/x"}), "advertising")
        self.assertEqual(classify_item({"title": "Creator subscription", "link": "https://www.producthunt.com/x"}), "monetization")

    def test_rejects_easy_money_claims(self):
        self.assertFalse(is_editorially_safe({"title": "Ingreso garantizado y dinero fácil"}, "monetization"))

    def test_ranking_adds_pillar_and_respects_limit(self):
        items = [
            {"title": "VALORANT final", "link": "https://dotesports.com/a"},
            {"title": "AI hardware", "link": "https://techcrunch.com/b"},
            {"title": "Brand campaign", "link": "https://blog.hubspot.com/c"},
        ]
        ranked = rank_articles_by_strategy(items, limit=2, rng=random.Random(7))
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all("pillar" in item for item in ranked))

    def test_planner_carries_pillar(self):
        plan = build_editorial_plan({"title": "New creator tool", "pillar": "technology"})
        self.assertEqual(plan["pillar"], "technology")
        self.assertEqual(plan["style_family"], "reel_tech")


if __name__ == "__main__":
    unittest.main()
