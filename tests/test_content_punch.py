import unittest

from media_factory.audience_intelligence import build_audience_experiment
from media_factory.content_punch import (
    build_content_punch,
    validate_content_punch,
)
from media_factory.models import Candidate


class ContentPunchTests(unittest.TestCase):
    def test_plan_contains_the_four_required_elements(self):
        candidate = Candidate(
            title="Google estudia cómo cambia el trabajo con IA",
            summary=(
                "El estudio analiza interacciones agregadas para identificar "
                "cómo colaboran las personas con la IA."
            ),
            source_url="https://example.com/google-ai",
            territory="ai_innovation_future",
            signals={
                "conversation_potential": 0.9,
                "angle_originality": 0.9,
            },
        )
        experiment = build_audience_experiment(candidate)
        plan = build_content_punch(candidate, experiment)

        self.assertTrue(plan["hook"])
        self.assertTrue(plan["concrete_value"])
        self.assertTrue(plan["tension_question"])
        self.assertTrue(plan["expected_action"])
        self.assertTrue(plan["gate_passed"])
        self.assertEqual(validate_content_punch(plan), [])
        self.assertFalse(plan["publishing_enabled"])

    def test_strong_tone_requires_two_real_signals(self):
        one_signal_only = Candidate(
            title="Una herramienta anuncia una nueva función",
            summary="La fuente describe una función nueva.",
            source_url="https://example.com/tool",
            territory="ai_innovation_future",
            signals={
                "conversation_potential": 0.95,
                "angle_originality": 0.4,
            },
        )
        experiment = build_audience_experiment(one_signal_only)
        plan = build_content_punch(one_signal_only, experiment)
        self.assertEqual(plan["tone"], "analytical")
        self.assertNotIn("ESTO CAMBIA LA CONVERSACIÓN", plan["hook"])

    def test_missing_summary_cannot_reuse_title_as_concrete_value(self):
        candidate = Candidate(
            title="LYON anuncia un evento competitivo en Ciudad de México",
            source_url="https://example.com/event",
            territory="gaming_esports",
        )
        experiment = build_audience_experiment(candidate)
        plan = build_content_punch(candidate, experiment)
        self.assertEqual(plan["concrete_value"], "")
        self.assertEqual(plan["evidence_origin"], "candidate.summary")
        self.assertFalse(plan["gate_passed"])
        self.assertIn(
            "missing_punch_field:concrete_value",
            validate_content_punch(plan),
        )

    def test_gate_rejects_missing_action(self):
        candidate = Candidate(
            title="Historia",
            summary="Resumen verificado.",
            source_url="https://example.com/story",
            territory="brands_activations",
        )
        experiment = build_audience_experiment(candidate)
        plan = build_content_punch(candidate, experiment)
        plan["expected_action"] = ""
        self.assertIn(
            "missing_punch_field:expected_action",
            validate_content_punch(plan),
        )

    def test_atlas_uses_contextual_hook_and_verified_numeric_value(self):
        candidate = Candidate(
            title="Google presenta AI & Economy ATLAS",
            summary=(
                "El estudio analiza 15 millones de interacciones agregadas "
                "en más de 150 países para entender cómo cambia el trabajo "
                "con IA."
            ),
            source_url="https://example.com/atlas",
            territory="ai_innovation_future",
            signals={
                "conversation_potential": 0.9,
                "angle_originality": 0.9,
            },
        )
        experiment = build_audience_experiment(candidate)
        plan = build_content_punch(candidate, experiment)

        self.assertEqual(plan["hook"], "¿LA IA TE POTENCIA O TE REEMPLAZA?")
        self.assertEqual(
            plan["concrete_value"],
            "15 MILLONES DE INTERACCIONES · MÁS DE 150 PAÍSES",
        )
        self.assertEqual(plan["evidence_origin"], "candidate.summary")


if __name__ == "__main__":
    unittest.main()
