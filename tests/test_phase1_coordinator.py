import json
import tempfile
import unittest
from pathlib import Path

from phase1_coordinator import (
    build_platform_readiness,
    run_coordinator,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase1CoordinatorTests(unittest.TestCase):
    def test_platform_readiness_reports_presence_without_secret_values(self):
        environment = {
            "THREADS_USER_ACCESS_TOKEN_CONFIGURED": "true",
            "THREADS_USER_ID_CONFIGURED": "true",
            "IG_ACCESS_TOKEN_CONFIGURED": "true",
            "IG_USER_ID_CONFIGURED": "false",
        }

        report = build_platform_readiness(environment)

        self.assertEqual(
            report["platforms"]["threads"]["status"],
            "configured_not_validated",
        )
        self.assertEqual(
            report["platforms"]["instagram"]["status"], "incomplete"
        )
        self.assertEqual(report["platforms"]["facebook"]["status"], "unknown")
        rendered = json.dumps(report)
        self.assertNotIn("access_token", rendered.lower())
        self.assertFalse(report["external_requests_attempted"])

    def test_coordinator_is_safe_and_creates_all_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            queue_path = output / "editorial.json"
            safety_path = output / "safety.json"
            readiness_path = output / "readiness.json"
            health_path = output / "health.json"

            health = run_coordinator(
                config_path=ROOT / "config" / "editorial_v1.json",
                input_path=ROOT / "fixtures" / "candidates.json",
                queue_output=queue_path,
                safety_output=safety_path,
                readiness_output=readiness_path,
                health_output=health_path,
                environment={},
            )

            self.assertTrue(health["healthy"])
            self.assertEqual(health["factory"]["publication_count"], 0)
            self.assertFalse(health["publishing_attempted"])
            self.assertFalse(health["external_writes_attempted"])
            self.assertFalse(health["paid_generation_attempted"])
            for path in (
                queue_path,
                safety_path,
                readiness_path,
                health_path,
            ):
                self.assertTrue(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
