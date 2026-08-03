import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tools.run_scheduled_publications import due_items, execute_queue, load_queue


class ScheduledPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        video = self.root / "approved.mp4"
        video.write_bytes(b"approved")
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        manifest = {
            "schema": "supervised_meta_publication_v1",
            "slug": "scheduled-test",
            "platform": "instagram",
            "video_path": "approved.mp4",
            "video_sha256": digest,
            "caption": "Texto aprobado",
            "approval_id": "approval-123",
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.queue_path = self.root / "queue.json"
        self.queue_path.write_text(json.dumps({
            "schema": "scheduled_publication_queue_v1",
            "timezone": "America/Bogota",
            "items": [{
                "content_id": "content-1",
                "manifest_path": "manifest.json",
                "approval_id": "approval-123",
                "publish_at": "2026-08-03T12:30:00-05:00",
                "status": "approved",
                "enabled": True,
            }],
        }), encoding="utf-8")
        self.env = {
            "AWS_ACCESS_KEY_ID": "configured",
            "AWS_SECRET_ACCESS_KEY": "configured",
            "R2_ENDPOINT_URL": "https://r2.example",
            "BUCKET_NAME": "bucket",
            "R2_PUBLIC_BASE_URL": "https://cdn.example",
            "IG_USER_ID": "ig-id",
            "IG_ACCESS_TOKEN": "ig-token",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_due_selection_excludes_publishing_and_published(self):
        queue = load_queue(self.queue_path)
        now = datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc)
        self.assertEqual(len(due_items(queue, {"items": {}}, now=now)), 1)
        for status in ("publishing", "published"):
            state = {"items": {"content-1": {"status": status}}}
            self.assertEqual(due_items(queue, state, now=now), [])

    def test_dry_run_validates_due_item_without_remote_state_write(self):
        with patch("tools.run_scheduled_publications.save_remote_state") as save:
            report = execute_queue(
                self.queue_path,
                repository_root=self.root,
                environment=self.env,
                now=datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc),
            )
        save.assert_not_called()
        self.assertEqual(report["due_count"], 1)
        self.assertTrue(report["results"][0]["dry_run"])

    def test_live_requires_both_production_gates(self):
        with self.assertRaisesRegex(RuntimeError, "production_not_armed"):
            execute_queue(
                self.queue_path,
                repository_root=self.root,
                environment=self.env,
                live=True,
            )
        armed = {**self.env, "PRODUCTION_ARMED": "true"}
        with self.assertRaisesRegex(RuntimeError, "scheduled_publishing_not_armed"):
            execute_queue(
                self.queue_path,
                repository_root=self.root,
                environment=armed,
                live=True,
            )

    def test_queue_and_manifest_approval_must_match(self):
        payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        payload["items"][0]["approval_id"] = "different"
        self.queue_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "approval_mismatch"):
            execute_queue(
                self.queue_path,
                repository_root=self.root,
                environment=self.env,
                now=datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc),
            )


if __name__ == "__main__":
    unittest.main()
