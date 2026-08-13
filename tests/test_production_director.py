import unittest
from pathlib import Path

from media_factory.production_director import build_production_request, load_cast


ROOT = Path(__file__).resolve().parents[1]


class ProductionDirectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cast = load_cast(ROOT / "config/virtual_cast_v2.json")
        cls.base = {
            "title": "Una historia verificada",
            "source_url": "https://example.com/source",
            "factual_summary": "Resumen con hechos comprobados.",
            "script": "Guion breve y comprensible.",
            "duration_seconds": 22,
        }

    def test_routes_content_types_to_their_character(self):
        cases = {"noticia": "nova", "artículo": "nova", "dato": "nova", "cuento": "rami", "relato": "rami", "chisme": "joseverso", "humor": "joseverso"}
        for content_type, expected in cases.items():
            with self.subTest(content_type=content_type):
                request = build_production_request({**self.base, "content_type": content_type}, self.cast, {})
                self.assertEqual(request.character_id, expected)

    def test_nova_and_rami_are_blocked_until_spanish_voice_exists(self):
        env = {"HEYGEN_NOVA_GROUP_ID": "group", "HEYGEN_NOVA_VOICE_ID": "voice"}
        request = build_production_request({**self.base, "content_type": "noticia"}, self.cast, env)
        self.assertIn("voice_language_not_ready", request.blockers)
        self.assertEqual(request.state, "blocked")

    def test_joseverso_can_pass_identity_checks_but_render_stays_disabled(self):
        env = {"HEYGEN_JOSEVERSO_GROUP_ID": "group", "HEYGEN_JOSEVERSO_VOICE_ID": "voice"}
        request = build_production_request({**self.base, "content_type": "chisme"}, self.cast, env)
        self.assertNotIn("voice_language_not_ready", request.blockers)
        self.assertEqual(request.state, "blocked")
        self.assertFalse(request.external_actions_enabled)

    def test_missing_evidence_is_blocked(self):
        request = build_production_request({"content_type": "dato"}, self.cast, {})
        self.assertIn("missing_source_url", request.blockers)
        self.assertIn("missing_script", request.blockers)


if __name__ == "__main__":
    unittest.main()
