import random
import unittest

from content_strategy import (
    DEFAULT_WEIGHTS,
    classify_item,
    commercial_angles_for_pillar,
    is_editorially_safe,
    rank_articles_by_strategy,
)
from editorial_planner import build_editorial_plan


class ContentStrategyTests(unittest.TestCase):
    def test_classifies_all_culture_pillars(self):
        cases = [
            ({"title": "VALORANT final", "link": "https://dotesports.com/x"}, "gaming"),
            ({"title": "New AI creator tool", "link": "https://blog.google/x"}, "technology"),
            ({"title": "Campaign strategy", "link": "https://blog.hubspot.com/x"}, "advertising"),
            ({"title": "New streetwear drop", "link": "https://hypebeast.com/x"}, "fashion"),
            ({"title": "Chef opens digital restaurant", "link": "https://www.eater.com/x"}, "gastronomy"),
            ({"title": "Digital home experience", "link": "https://www.dezeen.com/x"}, "lifestyle"),
            ({"title": "Luxury watch launch", "link": "https://robbreport.com/x"}, "luxury"),
            ({"title": "Creator subscription", "link": "https://www.producthunt.com/x"}, "monetization"),
        ]
        for item, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_item(item), expected)

    def test_weights_form_a_complete_mix(self):
        self.assertEqual(sum(DEFAULT_WEIGHTS.values()), 100)
        self.assertGreater(DEFAULT_WEIGHTS["gaming"] + DEFAULT_WEIGHTS["technology"], 50)

    def test_rejects_easy_money_claims(self):
        self.assertFalse(is_editorially_safe({"title": "Ingreso garantizado y dinero fácil"}, "monetization"))

    def test_ranking_adds_pillar_and_commercial_angles(self):
        items = [
            {"title": "VALORANT final", "link": "https://dotesports.com/a"},
            {"title": "AI hardware", "link": "https://techcrunch.com/b"},
            {"title": "Brand campaign", "link": "https://blog.hubspot.com/c"},
        ]
        ranked = rank_articles_by_strategy(items, limit=2, rng=random.Random(7))
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all("pillar" in item for item in ranked))
        self.assertTrue(all(item["commercial_angles"] for item in ranked))

    def test_commercial_angles_support_sales(self):
        self.assertIn("lead_calificado", commercial_angles_for_pillar("luxury"))
        self.assertIn("transmisión_en_vivo", commercial_angles_for_pillar("gaming"))
        self.assertIn("live_shopping", commercial_angles_for_pillar("fashion"))

    def test_planner_carries_every_visual_family(self):
        expected = {
            "technology": "reel_tech",
            "fashion": "reel_style",
            "gastronomy": "reel_food",
            "lifestyle": "reel_life",
            "luxury": "reel_luxury",
        }
        for pillar, family in expected.items():
            with self.subTest(pillar=pillar):
                plan = build_editorial_plan({"title": "New culture signal", "pillar": pillar})
                self.assertEqual(plan["pillar"], pillar)
                self.assertEqual(plan["style_family"], family)


if __name__ == "__main__":
    unittest.main()
