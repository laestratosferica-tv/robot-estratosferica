import unittest
import json
from pathlib import Path

from operations_safety import build_safety_report


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
OPERATIONS = json.loads(
    (ROOT / "config" / "operations_v1.json").read_text(encoding="utf-8")
)
PRODUCTION_WORKFLOWS = (
    "editorial.yml",
    "publisher.yml",
    "media_engine.yml",
    "ugc.yml",
)
LEGACY_GATE = "vars.LEGACY_WORKFLOWS_ARMED == 'true'"
PRODUCTION_GATE = "vars.PRODUCTION_ARMED == 'true'"


class WorkflowSafetyTests(unittest.TestCase):
    def test_legacy_publishers_require_explicit_production_arm(self):
        for filename in PRODUCTION_WORKFLOWS:
            with self.subTest(workflow=filename):
                content = (WORKFLOWS / filename).read_text(encoding="utf-8")
                self.assertIn(LEGACY_GATE, content)
                self.assertIn(PRODUCTION_GATE, content)

    def test_editorial_manual_publish_defaults_are_off(self):
        content = (WORKFLOWS / "editorial.yml").read_text(encoding="utf-8")
        self.assertIn('dry_run:\n        description: "Simular (no publica en redes)"', content)
        self.assertIn('publish_ig:\n        description: "Publicar en Instagram"', content)
        self.assertIn('publish_threads:\n        description: "Publicar en Threads"', content)
        self.assertIn('DRY_RUN: "true"', content)
        self.assertNotIn('ENABLE_FB_PUBLISH: "true"', content)

    def test_ugc_workflow_has_one_name_and_no_implicit_push(self):
        content = (WORKFLOWS / "ugc.yml").read_text(encoding="utf-8")
        self.assertEqual(
            sum(line.startswith("name:") for line in content.splitlines()),
            1,
        )
        self.assertNotIn("push:", content)

    def test_quarantined_workflows_have_no_schedule_and_require_gate(self):
        for filename, role in OPERATIONS["workflow_inventory"].items():
            if role != "legacy_quarantine":
                continue
            with self.subTest(workflow=filename):
                content = (WORKFLOWS / filename).read_text(encoding="utf-8")
                self.assertNotIn("schedule:", content)
                self.assertIn(LEGACY_GATE, content)

    def test_threads_defaults_cannot_auto_publish(self):
        accounts = json.loads(
            (ROOT / "accounts.json").read_text(encoding="utf-8")
        )["accounts"]
        for account in accounts:
            with self.subTest(account=account["account_id"]):
                threads = account["threads"]
                self.assertFalse(threads["auto_post"])
                self.assertEqual(threads["auto_post_limit"], 0)
                self.assertTrue(threads["dry_run"])

    def test_operations_inventory_matches_repository(self):
        actual = {path.name for path in WORKFLOWS.glob("*.yml")}
        self.assertEqual(actual, set(OPERATIONS["workflow_inventory"]))

    def test_safety_report_is_green(self):
        report = build_safety_report()
        self.assertTrue(report["safe"], report["errors"])
        self.assertEqual(report["scheduled_workflows"], [])
        self.assertFalse(report["publishing_enabled"])
        self.assertFalse(report["external_writes_enabled"])
        self.assertFalse(report["paid_generation_enabled"])

    def test_safe_coordinator_uses_presence_flags_not_secret_values(self):
        content = (
            WORKFLOWS / "factory-v1-dry-run.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("python phase1_coordinator.py", content)
        self.assertIn(
            "THREADS_USER_ACCESS_TOKEN_CONFIGURED:", content
        )
        self.assertNotIn(
            "THREADS_USER_ACCESS_TOKEN: ${{ secrets.", content
        )
        self.assertIn("artifacts/coordinator-health.json", content)
        self.assertIn("artifacts/platform-readiness.json", content)

    def test_threads_diagnostic_does_not_rotate_by_default(self):
        content = (
            WORKFLOWS / "threads-auth-check.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("default: validate_existing", content)
        self.assertIn("if: inputs.mode == 'validate_existing'", content)
        self.assertEqual(
            content.count("if: inputs.mode == 'prepare_rotation'"), 2
        )


if __name__ == "__main__":
    unittest.main()
