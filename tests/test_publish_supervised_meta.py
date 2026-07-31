import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.publish_supervised_meta import load_manifest, run


class SupervisedMetaPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "approved.mp4"
        self.video.write_bytes(b"approved-video")
        self.video_hash = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.environment = {
            "AWS_ACCESS_KEY_ID": "configured",
            "AWS_SECRET_ACCESS_KEY": "configured",
            "R2_ENDPOINT_URL": "https://r2.example",
            "BUCKET_NAME": "bucket",
            "R2_PUBLIC_BASE_URL": "https://cdn.example",
            "IG_USER_ID": "ig-id",
            "IG_ACCESS_TOKEN": "ig-secret",
        }

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, **changes):
        payload = {
            "schema": "supervised_meta_publication_v1",
            "slug": "piece-001",
            "platform": "instagram",
            "video_path": "approved.mp4",
            "video_sha256": self.video_hash,
            "caption": "Copy aprobado",
            "approval_id": "approval-001",
        }
        payload.update(changes)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_dry_run_validates_without_external_writes(self):
        with patch("tools.publish_supervised_meta.upload_public_video") as upload:
            receipt = run(
                self.manifest(),
                repository_root=self.root,
                environment=self.environment,
            )
        upload.assert_not_called()
        self.assertTrue(receipt["dry_run"])
        self.assertFalse(receipt["publishing_attempted"])
        self.assertFalse(receipt["published"])
        self.assertNotIn("ig-secret", json.dumps(receipt))

    def test_rejects_multiple_or_unknown_platforms(self):
        with self.assertRaisesRegex(ValueError, "single_meta_network"):
            load_manifest(self.manifest(platform=["instagram", "facebook"]))

    def test_rejects_changed_video(self):
        path = self.manifest(video_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            run(
                path,
                repository_root=self.root,
                environment=self.environment,
            )

    def test_live_mode_requires_exact_approval_and_production_arm(self):
        path = self.manifest()
        with self.assertRaisesRegex(RuntimeError, "production_not_armed"):
            run(
                path,
                repository_root=self.root,
                environment=self.environment,
                dry_run=False,
            )
        armed = {**self.environment, "PRODUCTION_ARMED": "true"}
        with self.assertRaisesRegex(RuntimeError, "approval_mismatch"):
            run(path, repository_root=self.root, environment=armed, dry_run=False)

    def test_simulation_manifest_can_never_publish(self):
        path = self.manifest(simulation_only=True)
        armed = {
            **self.environment,
            "PRODUCTION_ARMED": "true",
            "PUBLICATION_APPROVAL_ID": "approval-001",
        }
        with self.assertRaisesRegex(RuntimeError, "simulation_only"):
            run(path, repository_root=self.root, environment=armed, dry_run=False)


if __name__ == "__main__":
    unittest.main()
