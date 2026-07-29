import unittest

from media_factory.models import Candidate
from media_factory.retention import (
    build_retention_plan,
    load_retention_playbook,
    validate_retention_plan,
    validate_retention_playbook,
)


class RetentionPlaybookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.playbook = load_retention_playbook()

    def test_permanent_playbook_is_complete_and_safe(self) -> None:
        self.assertEqual(validate_retention_playbook(self.playbook), [])
        self.assertEqual(
            set(self.playbook["formats"]),
            {"short_video", "photo", "carousel", "long_video", "text_post"},
        )

    def test_target_feels_native_not_school_like(self) -> None:
        identity = self.playbook["identity"]
        self.assertEqual(
            identity["relationship"],
            "persona_del_mismo_mundo_no_profesor",
        )
        self.assertIn("clase", identity["forbidden_feeling"])
        self.assertIn("tarea", identity["forbidden_feeling"])

    def test_short_video_is_fast_and_measurable(self) -> None:
        rules = self.playbook["formats"]["short_video"]
        self.assertEqual(rules["target_duration_seconds"], [8, 15])
        self.assertLessEqual(rules["hook_deadline_seconds"], 1)
        self.assertIn("retencion_1s", rules["metrics"])
        self.assertIn("feedback_negativo", rules["metrics"])

    def test_generated_plan_passes_automatic_gate(self) -> None:
        plan = build_retention_plan(
            Candidate(
                title="Cambio competitivo",
                source_url="https://example.com/story",
                territory="gaming_esports",
            )
        )
        self.assertEqual(validate_retention_plan(plan), [])
        self.assertTrue(plan["gate_passed"])

    def test_addiction_objective_is_rejected(self) -> None:
        plan = build_retention_plan(
            Candidate(
                title="Cambio competitivo",
                source_url="https://example.com/story",
                territory="gaming_esports",
            )
        )
        unsafe = {**plan, "ethical_objective": "adiccion"}
        self.assertIn(
            "unsafe_retention_objective",
            validate_retention_plan(unsafe),
        )

    def test_missing_ethical_rule_is_rejected(self) -> None:
        changed = {
            **self.playbook,
            "ethical_retention": {
                **self.playbook["ethical_retention"],
                "prohibited_objectives": ["enganio"],
            },
        }
        self.assertIn(
            "incomplete_ethical_retention_policy",
            validate_retention_playbook(changed),
        )


if __name__ == "__main__":
    unittest.main()
