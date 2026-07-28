import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "config" / "content_strategy_v1.json"


class ContentStrategyV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.strategy = json.loads(STRATEGY_PATH.read_text(encoding="utf-8"))

    def test_editorial_pillars_keep_gaming_as_the_core(self) -> None:
        pillars = self.strategy["editorial_pillars"]
        self.assertEqual(sum(pillars.values()), 100)
        self.assertGreaterEqual(pillars["gaming_esports"], 50)
        self.assertIn("technology_innovation", pillars)
        self.assertIn("fashion_lifestyle", pillars)
        self.assertIn("gastronomy_experiences", pillars)

    def test_every_product_has_a_purpose_and_measurement_path(self) -> None:
        required = {
            "id",
            "name",
            "purpose",
            "funnel_stage",
            "community_action",
            "primary_metrics",
            "commercial_paths",
            "rights_requirement",
        }
        products = self.strategy["content_products"]
        self.assertGreaterEqual(len(products), 8)
        for product in products:
            self.assertTrue(required <= product.keys())
            self.assertTrue(product["primary_metrics"])
            self.assertTrue(product["commercial_paths"])

    def test_strategy_includes_community_broadcasts_and_owned_events(self) -> None:
        product_ids = {
            product["id"] for product in self.strategy["content_products"]
        }
        self.assertIn("comunidad_decide", product_ids)
        self.assertIn("arena_estratosferica", product_ids)
        self.assertIn("torneo_estratosferico", product_ids)
        self.assertIn("informe_de_audiencia", product_ids)

    def test_unverified_retransmission_and_reuploads_are_blocked(self) -> None:
        rights = self.strategy["rights"]
        self.assertIn("authorized_free", rights["allowed_paths"])
        self.assertIn("licensed_paid", rights["allowed_paths"])
        self.assertIn("original_owned", rights["allowed_paths"])
        self.assertIn(
            "retransmit_unverified_event",
            rights["blocked_without_written_clearance"],
        )
        self.assertIn(
            "reupload_third_party_clip",
            rights["blocked_without_written_clearance"],
        )
        self.assertTrue(rights["rights_record_required_before_broadcast"])

    def test_success_cannot_be_based_only_on_views(self) -> None:
        measurement = self.strategy["measurement"]
        self.assertTrue(measurement["views_only_success_is_forbidden"])
        self.assertIn("community", measurement["dimensions"])
        self.assertIn("ownership", measurement["dimensions"])
        self.assertIn("business", measurement["dimensions"])
        self.assertFalse(measurement["automatic_strategy_changes_enabled"])

    def test_all_external_actions_remain_human_controlled(self) -> None:
        safety = self.strategy["safety"]
        self.assertFalse(safety["publishing_enabled"])
        self.assertFalse(safety["broadcasting_enabled"])
        self.assertFalse(safety["automatic_commercial_outreach_enabled"])
        self.assertTrue(safety["human_approval_required"])


if __name__ == "__main__":
    unittest.main()
