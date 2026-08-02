import unittest

from commerce_platform_routing import (
    build_amazon_routes,
    validate_amazon_distribution,
)


class CommercePlatformRoutingTests(unittest.TestCase):
    def test_amazon_routes_cover_every_target_platform(self):
        routes = build_amazon_routes("https://www.amazon.com/dp/EXAMPLE?tag=test-20")
        self.assertEqual(
            set(routes),
            {
                "instagram_reel",
                "instagram_story",
                "facebook",
                "threads",
                "youtube_short",
            },
        )
        self.assertEqual(validate_amazon_distribution(routes), [])

    def test_instagram_reel_and_youtube_do_not_promise_clickable_caption(self):
        routes = build_amazon_routes("https://www.amazon.com/dp/EXAMPLE?tag=test-20")
        self.assertFalse(routes["instagram_reel"].requires_clickable_link)
        self.assertFalse(routes["youtube_short"].requires_clickable_link)
        self.assertTrue(routes["instagram_story"].requires_clickable_link)

    def test_rejects_insecure_affiliate_link(self):
        with self.assertRaises(ValueError):
            build_amazon_routes("http://www.amazon.com/dp/EXAMPLE")

    def test_incomplete_distribution_is_rejected(self):
        routes = build_amazon_routes("https://www.amazon.com/dp/EXAMPLE?tag=test-20")
        routes.pop("threads")
        self.assertEqual(
            validate_amazon_distribution(routes),
            ["Faltan rutas: threads"],
        )


if __name__ == "__main__":
    unittest.main()
