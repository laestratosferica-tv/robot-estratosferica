import json
import tempfile
import unittest
from pathlib import Path

from phase1_coordinator import (
    CoordinatorError,
    build_platform_readiness,
    run_coordinator,
)
from phase1_acceptance import run_acceptance


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
                input_path=(
                    ROOT / "fixtures" / "real_candidates_2026-07-25.json"
                ),
                sources_path=ROOT / "config" / "sources_v1.json",
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
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            self.assertEqual(queue["schema_version"], "review_queue_v1")
            self.assertTrue(queue["human_approval_required"])
            self.assertTrue(
                all(
                    item["review"]["status"]
                    == "pending_human_approval"
                    for item in queue["items"]
                )
            )
            self.assertEqual(
                health["factory"]["strategy_classified_count"],
                health["factory"]["candidate_count"],
            )
            self.assertTrue(
                all(
                    item["review"]["strategy"]["content_product_id"]
                    for item in queue["items"]
                )
            )
            self.assertTrue(
                all(
                    item["review"]["strategy"]["requires_human_review"]
                    for item in queue["items"]
                )
            )
            self.assertTrue(health["source_registry_enforced"])
            self.assertEqual(health["cost"]["billable_operations"], 0)
            self.assertEqual(health["cost"]["measured_cost_usd"], 0.0)
            for path in (
                queue_path,
                safety_path,
                readiness_path,
                health_path,
            ):
                self.assertTrue(path.exists(), path)

    def test_coordinator_rejects_missing_source_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(
                CoordinatorError, "verified source registry"
            ):
                run_coordinator(
                    config_path=ROOT / "config" / "editorial_v1.json",
                    input_path=ROOT / "fixtures" / "candidates.json",
                    queue_output=output / "queue.json",
                    safety_output=output / "safety.json",
                    readiness_output=output / "readiness.json",
                    health_output=output / "health.json",
                    environment={},
                )

    def test_acceptance_requires_five_stable_safe_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_acceptance(
                output_path=Path(directory) / "acceptance.json",
                environment={},
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["required_consecutive_runs"], 5)
            self.assertEqual(report["healthy_consecutive_runs"], 5)
            self.assertTrue(report["stable_queue"])
            self.assertEqual(report["unique_queue_digests"], 1)
            self.assertEqual(report["publication_count"], 0)
            self.assertEqual(report["duplicate_publication_count"], 0)
            self.assertEqual(report["billable_operations"], 0)
            self.assertEqual(report["measured_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
