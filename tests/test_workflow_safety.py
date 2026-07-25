import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PRODUCTION_WORKFLOWS = (
    "editorial.yml",
    "publisher.yml",
    "media_engine.yml",
    "ugc.yml",
)
KILL_SWITCH = "if: vars.PRODUCTION_ARMED == 'true'"


class WorkflowSafetyTests(unittest.TestCase):
    def test_legacy_publishers_require_explicit_production_arm(self):
        for filename in PRODUCTION_WORKFLOWS:
            with self.subTest(workflow=filename):
                content = (WORKFLOWS / filename).read_text(encoding="utf-8")
                self.assertIn(KILL_SWITCH, content)

    def test_editorial_manual_publish_defaults_are_off(self):
        content = (WORKFLOWS / "editorial.yml").read_text(encoding="utf-8")
        self.assertNotIn('default: "true"', content)


if __name__ == "__main__":
    unittest.main()
