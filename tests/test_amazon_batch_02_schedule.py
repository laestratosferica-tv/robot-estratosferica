import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tools.publish_approved_social_post import load_manifest as load_social
from tools.publish_approved_youtube_short import load_manifest as load_youtube
from tools.publish_supervised_meta import load_manifest as load_meta, validate_manifest
from tools.run_scheduled_publications import due_items, load_queue, validate_item


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "config/scheduled_publications_v1.json"
PRODUCTS = {
    "bengoo-g9000-v1": "2026-08-06T17:30:00-05:00",
    "alpha-grillers-v1": "2026-08-09T16:30:00-05:00",
    "tapo-c100-v1": "2026-08-12T17:30:00-05:00",
    "astroai-c2-v1": "2026-08-18T17:30:00-05:00",
}


class AmazonBatch02ScheduleTests(unittest.TestCase):
    def test_four_routes_per_product_have_exact_bogota_schedule(self):
        queue = load_queue(QUEUE)
        for product, scheduled in PRODUCTS.items():
            items = [item for item in queue["items"] if item["content_id"].startswith(product)]
            self.assertEqual(len(items), 4)
            self.assertTrue(all(item["publish_at"] == scheduled for item in items))
            before = datetime.fromisoformat(scheduled).astimezone(timezone.utc)
            before = before.replace(microsecond=0)
            due = due_items(queue, {"items": {}}, now=before)
            self.assertEqual(len([item for item in due if item["content_id"].startswith(product)]), 4)

    def test_all_automatic_manifests_and_commercial_evidence_validate(self):
        queue = load_queue(QUEUE)
        for product in PRODUCTS:
            items = [item for item in queue["items"] if item["content_id"].startswith(product)]
            for item in items:
                path, manifest = validate_item(item, ROOT)
                if manifest["schema"] == "supervised_meta_publication_v1":
                    validate_manifest(load_meta(path), repository_root=ROOT)
                elif manifest["schema"] == "approved_social_post_v1":
                    load_social(path, ROOT)
                else:
                    load_youtube(path, ROOT)

    def test_stories_are_supervised_and_led_ball_is_excluded(self):
        queue_text = QUEUE.read_text(encoding="utf-8").lower()
        self.assertNotIn("led-ball", queue_text)
        for product in PRODUCTS:
            story = json.loads((ROOT / f"artifacts/publication-manifests/{product}-instagram-story.json").read_text())
            self.assertEqual(story["execution"], "native_supervised_not_in_automatic_queue")
            self.assertNotIn(f"{product}-instagram-story.json", queue_text)


if __name__ == "__main__":
    unittest.main()
