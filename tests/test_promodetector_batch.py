import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_promodetector_batch import DEFAULT_MANIFEST, validate


class PromoDetectorBatchTests(unittest.TestCase):
    def test_current_batch_is_ready(self):
        self.assertEqual(validate(), [])

    def test_publication_permission_fails_closed(self):
        data = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        data["publication_allowed"] = True
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "batch.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            self.assertIn(
                "publication_allowed debe permanecer en false durante revisión",
                validate(manifest),
            )


if __name__ == "__main__":
    unittest.main()
