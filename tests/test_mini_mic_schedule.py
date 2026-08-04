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


class MiniMicScheduleTests(unittest.TestCase):
    def test_four_automatic_routes_are_scheduled_at_2300_bogota(self):
        queue = load_queue(QUEUE)
        items = [item for item in queue["items"] if item["content_id"].startswith("mini-mic-v1-")]
        self.assertEqual(len(items), 4)
        self.assertTrue(all(item["publish_at"] == "2026-08-03T23:00:00-05:00" for item in items))
        before = datetime.fromisoformat("2026-08-03T22:59:59-05:00").astimezone(timezone.utc)
        at_time = datetime.fromisoformat("2026-08-03T23:00:00-05:00").astimezone(timezone.utc)
        self.assertFalse(any(item["content_id"].startswith("mini-mic") for item in due_items(queue, {"items": {}}, now=before)))
        self.assertEqual(
            len([item for item in due_items(queue, {"items": {}}, now=at_time) if item["content_id"].startswith("mini-mic")]),
            4,
        )

    def test_approved_video_and_all_automatic_manifests_validate(self):
        queue = load_queue(QUEUE)
        items = [item for item in queue["items"] if item["content_id"].startswith("mini-mic-v1-")]
        for item in items:
            path, manifest = validate_item(item, ROOT)
            if manifest["schema"] == "supervised_meta_publication_v1":
                validate_manifest(load_meta(path), repository_root=ROOT)
            elif manifest["schema"] == "approved_social_post_v1":
                load_social(path, ROOT)
            else:
                load_youtube(path, ROOT)

    def test_story_route_is_supervised_and_not_queued(self):
        story_path = ROOT / "artifacts/publication-manifests/mini-mic-v1-instagram-story.json"
        story = json.loads(story_path.read_text(encoding="utf-8"))
        self.assertEqual(story["execution"], "native_supervised_not_in_automatic_queue")
        queue_text = QUEUE.read_text(encoding="utf-8")
        self.assertNotIn("mini-mic-v1-instagram-story.json", queue_text)


if __name__ == "__main__":
    unittest.main()
