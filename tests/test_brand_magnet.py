import unittest

from media_factory.brand_magnet import (
    build_private_concept,
    load_brand_magnet,
    qualify_brand_opportunity,
    validate_brand_magnet,
)


class BrandMagnetTests(unittest.TestCase):
    def test_configuration_is_complete_and_safe(self):
        config = load_brand_magnet()
        self.assertEqual(validate_brand_magnet(config), [])
        self.assertFalse(config["safety"]["automatic_outreach_enabled"])
        self.assertFalse(config["safety"]["publishing_enabled"])

    def test_ramo_like_opportunity_selects_community_quest(self):
        result = qualify_brand_opportunity({
            "category": "alimentos",
            "objective": "participación e investigación de comunidad",
            "gamer_fit": 1,
            "latam_relevance": 1,
            "community_value": 0.9,
            "measurable_objective": 0.8,
            "rights_readiness": 1,
            "commercial_readiness": 0.8,
        })
        self.assertEqual(result["status"], "qualified")
        self.assertEqual(result["offer_id"], "community_quest")
        self.assertFalse(result["automatic_outreach_enabled"])

    def test_weak_fit_stays_research_only(self):
        result = qualify_brand_opportunity({
            "category": "retail",
            "objective": "awareness",
            "gamer_fit": 0.1,
            "latam_relevance": 0.2,
        })
        self.assertEqual(result["status"], "research_only")
        self.assertTrue(result["requires_human_review"])

    def test_blocked_category_is_rejected(self):
        result = qualify_brand_opportunity({
            "category": "apuestas_no_autorizadas",
            "gamer_fit": 1,
        })
        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["offer_id"])

    def test_private_concept_never_implies_partnership(self):
        concept = build_private_concept(
            brand="Ramo",
            business_goal="Afinidad",
            gamer_tension="La publicidad interrumpe",
            concept_name="Checkpoint",
            mechanic="Reto comunitario",
            proof_plan=["retención", "participación"],
        )
        self.assertEqual(concept["status"], "private_unofficial_concept")
        self.assertIn("no implica relación", concept["disclaimer"].casefold())
        self.assertFalse(concept["publishing_enabled"])
        self.assertFalse(concept["outreach_enabled"])


if __name__ == "__main__":
    unittest.main()
