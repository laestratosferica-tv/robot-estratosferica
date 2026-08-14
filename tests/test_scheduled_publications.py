import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tools.run_scheduled_publications import (
    due_items,
    execute_queue,
    load_queue,
    publish_item,
)


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

    def test_dispatches_carousel_youtube_and_social_schemas(self):
        cases = (
            ("approved_carousel_publication_v1", "publish_carousel"),
            ("approved_youtube_short_publication_v1", "publish_youtube"),
            ("approved_social_post_v1", "publish_social_post"),
        )
        for schema, publisher in cases:
            with self.subTest(schema=schema), patch(
                f"tools.run_scheduled_publications.{publisher}"
            ) as mocked:
                mocked.return_value = {"published": False, "platforms": {}}
                publish_item(
                    self.root / "manifest.json",
                    {"schema": schema},
                    repository_root=self.root,
                    environment=self.env,
                    live=False,
                )
                mocked.assert_called_once()

    def test_commercial_item_requires_all_checks(self):
        payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        payload["items"][0]["commercial"] = True
        payload["items"][0]["commercial_checks"] = {
            "asset_final": False,
        }
        self.queue_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "commercial_publication_checks_incomplete"):
            load_queue(self.queue_path)

    def test_commercial_item_requires_resolver_evidence_path(self):
        payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        payload["items"][0].update({
            "commercial": True,
            "commercial_checks": {"asset_final": True},
        })
        self.queue_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "commercial_resolver_evidence_missing"):
            load_queue(self.queue_path)

    def test_fun_source_reused_on_different_days_is_blocked(self):
        evidence = {
            "source_page": "https://example.test/gameplay.webm",
            "license": "CC0-1.0",
        }
        (self.root / "source.json").write_text(json.dumps(evidence), encoding="utf-8")
        payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        payload["items"][0].update({
            "category": "contenido_divertido",
            "license_evidence_path": "source.json",
        })
        second = {
            **payload["items"][0],
            "content_id": "content-2",
            "publish_at": "2026-08-04T12:30:00-05:00",
        }
        payload["items"].append(second)
        self.queue_path.write_text(json.dumps(payload), encoding="utf-8")

        queue = load_queue(self.queue_path)
        now = datetime(2026, 8, 5, 18, 0, tzinfo=timezone.utc)

        self.assertEqual(due_items(queue, {"items": {}}, now=now), [])
        self.assertEqual(queue["repeat_guard"]["blocked_items"], 2)
        self.assertEqual(
            queue["repeat_guard"]["blocked_sources"],
            ["https://example.test/gameplay.webm"],
        )

    def test_fun_source_can_publish_to_multiple_platforms_same_day(self):
        evidence = {"source_page": "https://example.test/unique.webm"}
        (self.root / "source.json").write_text(json.dumps(evidence), encoding="utf-8")
        payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        payload["items"][0].update({
            "category": "contenido_divertido",
            "license_evidence_path": "source.json",
        })
        second = {**payload["items"][0], "content_id": "content-2"}
        payload["items"].append(second)
        self.queue_path.write_text(json.dumps(payload), encoding="utf-8")

        queue = load_queue(self.queue_path)
        now = datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc)

        self.assertEqual(len(due_items(queue, {"items": {}}, now=now)), 2)
        self.assertEqual(queue["repeat_guard"]["blocked_items"], 0)

    def test_one_failed_item_does_not_block_later_due_items(self):
        payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        second = {**payload["items"][0], "content_id": "content-2"}
        payload["items"].append(second)
        self.queue_path.write_text(json.dumps(payload), encoding="utf-8")
        armed = {
            **self.env,
            "PRODUCTION_ARMED": "true",
            "SCHEDULED_PUBLISHING_ARMED": "true",
        }
        with patch("tools.run_scheduled_publications.load_remote_state", return_value={"schema": "scheduled_publication_state_v1", "items": {}}), patch(
            "tools.run_scheduled_publications.save_remote_state"
        ), patch(
            "tools.run_scheduled_publications.publish_item",
            side_effect=[RuntimeError("provider_failed"), {"published": True, "platform": "instagram"}],
        ):
            report = execute_queue(
                self.queue_path,
                repository_root=self.root,
                environment=armed,
                now=datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc),
                live=True,
            )
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(len(report["results"]), 2)
        self.assertTrue(report["results"][1]["published"])


if __name__ == "__main__":
    unittest.main()
