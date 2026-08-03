import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.publish_approved_social_post import load_manifest, run


class ApprovedSocialPostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.asset = self.root / "approved.png"
        self.asset.write_bytes(b"approved-image")
        self.digest = hashlib.sha256(self.asset.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, **changes):
        payload = {
            "schema": "approved_social_post_v1",
            "slug": "post-test",
            "platform": "instagram",
            "post_type": "image",
            "approval_id": "approval-1",
            "text": "Texto aprobado",
            "asset_path": "approved.png",
            "asset_sha256": self.digest,
        }
        payload.update(changes)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_dry_run_validates_image_without_network(self):
        path = self.write_manifest()
        receipt = run(path, root=self.root, live=False, env={})
        self.assertFalse(receipt["published"])

    def test_text_link_requires_https(self):
        path = self.write_manifest(
            platform="facebook",
            post_type="text_link",
            link="http://example.com",
        )
        with self.assertRaisesRegex(ValueError, "https_link"):
            load_manifest(path, self.root)

    def test_interactive_story_stays_manual(self):
        path = self.write_manifest(
            post_type="story_image",
            interactive_sticker_required=True,
        )
        with self.assertRaisesRegex(ValueError, "native_manual_step"):
            load_manifest(path, self.root)

    def test_live_requires_matching_approval(self):
        path = self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, "production_approval_not_armed"):
            run(path, root=self.root, live=True, env={"PRODUCTION_ARMED": "true"})


if __name__ == "__main__":
    unittest.main()
