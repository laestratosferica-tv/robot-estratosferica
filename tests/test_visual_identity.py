import io
import unittest

from PIL import Image

from editorial_graphics import build_threads_card
from visual_identity import CATEGORY_DIRECTIONS, get_visual_direction


class VisualIdentityTests(unittest.TestCase):
    def test_every_pillar_has_a_distinct_layout(self):
        layouts = {
            pillar: direction["layout"]
            for pillar, direction in CATEGORY_DIRECTIONS.items()
        }
        self.assertEqual(len(layouts), 8)
        self.assertGreaterEqual(len(set(layouts.values())), 7)

    def test_brand_signature_survives_every_category(self):
        for pillar in CATEGORY_DIRECTIONS:
            with self.subTest(pillar=pillar):
                direction = get_visual_direction(pillar)
                self.assertEqual(direction["brand"]["short_mark"], "LETV")
                self.assertEqual(direction["brand"]["symbol"], "orbital_arc")
                self.assertEqual(len(direction["brand"]["core_gradient"]), 3)

    def test_card_renders_in_social_format_for_each_pillar(self):
        source = Image.new("RGB", (900, 900), "#34415C")
        payload = io.BytesIO()
        source.save(payload, format="JPEG")
        for pillar in CATEGORY_DIRECTIONS:
            with self.subTest(pillar=pillar):
                rendered = build_threads_card(
                    image_bytes=payload.getvalue(),
                    headline="Una historia distinta para cada universo",
                    badge_text="SEÑAL",
                    pillar=pillar,
                    trend_profile="sport_luxe_2026_q3",
                )
                card = Image.open(io.BytesIO(rendered))
                self.assertEqual(card.size, (1080, 1350))
                self.assertEqual(card.format, "PNG")

    def test_categories_change_background_structure_not_only_color(self):
        modes = {
            pillar: direction["background_mode"]
            for pillar, direction in CATEGORY_DIRECTIONS.items()
        }
        self.assertEqual(len(set(modes.values())), 8)
        self.assertEqual(CATEGORY_DIRECTIONS["gaming"]["background_mode"], "full_bleed")
        self.assertEqual(CATEGORY_DIRECTIONS["luxury"]["background_mode"], "gallery_window")
        self.assertNotEqual(
            CATEGORY_DIRECTIONS["technology"]["focal_anchor"],
            CATEGORY_DIRECTIONS["advertising"]["focal_anchor"],
        )

    def test_background_palette_is_derived_from_each_photo(self):
        outputs = []
        for source_color in ("#143B7A", "#C45C21"):
            source = Image.new("RGB", (900, 900), source_color)
            payload = io.BytesIO()
            source.save(payload, format="JPEG")
            rendered = build_threads_card(
                image_bytes=payload.getvalue(),
                headline="El fondo responde a la historia",
                badge_text="SEÑAL",
                pillar="fashion",
            )
            card = Image.open(io.BytesIO(rendered)).convert("RGB")
            outputs.append(card.getpixel((20, 700)))
        self.assertNotEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
