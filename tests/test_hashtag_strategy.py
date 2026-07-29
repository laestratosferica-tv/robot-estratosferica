import unittest

from media_factory.hashtag_strategy import replace_hashtags, select_hashtags


class HashtagStrategyTests(unittest.TestCase):
    def test_halo_uses_precise_latam_brand_set(self):
        tags = select_hashtags(
            game="Halo",
            topic="Operación Meteorite",
            intent="news",
        )
        self.assertEqual(
            tags,
            [
                "#Halo",
                "#MasterChief",
                "#OperacionMeteorite",
                "#NoticiasGaming",
                "#GamingLatam",
                "#LaEstratosferica",
            ],
        )

    def test_generic_filler_is_removed_and_limit_is_respected(self):
        tags = select_hashtags(
            game="Minecraft",
            extra_tags=["#FYP", "#Viral", "#Minecraft", "#Extra"],
        )
        self.assertLessEqual(len(tags), 6)
        self.assertNotIn("#FYP", tags)
        self.assertNotIn("#Viral", tags)
        self.assertEqual(len(tags), len({tag.casefold() for tag in tags}))

    def test_generated_hashtags_are_replaced(self):
        caption = replace_hashtags(
            "Gancho.\n\nFuente: https://example.com\n#viral #fyp",
            ["#Halo", "#GamingLatam"],
        )
        self.assertNotIn("#viral", caption)
        self.assertNotIn("#fyp", caption)
        self.assertTrue(caption.endswith("#Halo #GamingLatam"))


if __name__ == "__main__":
    unittest.main()
