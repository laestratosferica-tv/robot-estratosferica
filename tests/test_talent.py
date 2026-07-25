import json
import unittest
from pathlib import Path

from media_factory.talent import (
    load_talent_catalog,
    select_talent,
    validate_public_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "talent_v1.json"


class TalentTests(unittest.TestCase):
    def test_radar_uses_la_senal(self):
        catalog = load_talent_catalog(CATALOG_PATH)
        selection = select_talent(
            "gaming_esports", "radar_estratosferico", catalog
        )
        self.assertEqual(selection.character_id, "la_senal")

    def test_brand_play_uses_la_analista(self):
        catalog = load_talent_catalog(CATALOG_PATH)
        selection = select_talent(
            "brands_activations", "brand_play", catalog
        )
        self.assertEqual(selection.character_id, "la_analista")

    def test_jose_luis_stays_disabled_until_private_package_is_ready(self):
        catalog = load_talent_catalog(CATALOG_PATH)
        character = next(
            item
            for item in catalog["characters"]
            if item["id"] == "jose_luis_curador"
        )
        self.assertFalse(character["enabled"])

    def test_rejects_private_fields_in_public_catalog(self):
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        catalog["characters"][0]["face_reference"] = "private/photo.jpg"
        with self.assertRaises(ValueError):
            validate_public_catalog(catalog)


if __name__ == "__main__":
    unittest.main()
