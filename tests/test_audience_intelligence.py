import unittest

from media_factory.audience_intelligence import (
    PLATFORM_PLAYBOOKS,
    build_audience_experiment,
)
from media_factory.metrics import (
    calculate_performance_metrics,
    summarize_audience_learning,
)
from media_factory.models import Candidate


class AudienceIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.candidate = Candidate(
            title="La IA transforma el trabajo creativo en Latinoamérica",
            source_url="https://example.com/ai",
            territory="ai_innovation_future",
            signals={
                "conversation_potential": 0.9,
                "explanatory_value": 0.8,
            },
        )

    def test_experiment_is_stable_and_safe(self):
        first = build_audience_experiment(self.candidate)
        second = build_audience_experiment(self.candidate)
        self.assertEqual(first, second)
        self.assertFalse(first["publishing_enabled"])
        self.assertTrue(first["requires_human_review"])
        self.assertGreaterEqual(len(first["answer_options"]), 2)

    def test_every_platform_gets_a_distinct_native_plan(self):
        experiment = build_audience_experiment(self.candidate)
        self.assertEqual(
            set(experiment["platform_plans"]),
            {"threads", "instagram", "facebook", "youtube"},
        )
        for platform, plan in experiment["platform_plans"].items():
            self.assertIn(plan["format"], PLATFORM_PLAYBOOKS[platform]["formats"])
            self.assertEqual(plan["state"], "draft")
            self.assertFalse(plan["publishing_enabled"])

    def test_unsupported_api_poll_has_a_safe_fallback(self):
        experiment = build_audience_experiment(self.candidate)
        for plan in experiment["platform_plans"].values():
            self.assertFalse(plan["native_poll_api"])
            self.assertTrue(plan["manual_poll_surface"])
            self.assertGreaterEqual(len(plan["answer_options"]), 2)
            self.assertEqual(
                plan["poll_fallback"], "question_with_structured_options"
            )

    def test_views_alone_do_not_win_the_learning_score(self):
        viral_but_empty = calculate_performance_metrics({"views": 100000})
        smaller_but_useful = calculate_performance_metrics({
            "views": 1000,
            "comments": 80,
            "qualified_answers": 60,
            "shares": 40,
            "saves": 50,
            "follows": 20,
            "completion_rate": 0.7,
        })
        self.assertGreater(
            smaller_but_useful["learning_score"],
            viral_but_empty["learning_score"],
        )

    def test_learning_report_never_changes_strategy_automatically(self):
        report = summarize_audience_learning([
            {"experiment_id": "a", "views": 500, "comments": 20},
            {"experiment_id": "b", "views": 800, "shares": 30},
        ])
        self.assertEqual(report["mode"], "analysis_only")
        self.assertFalse(report["automatic_strategy_changes_enabled"])
        self.assertTrue(report["requires_human_review"])

    def test_question_follows_the_verified_story_context(self):
        atlas = Candidate(
            title="Google presenta AI & Economy ATLAS",
            summary=(
                "El estudio analiza 15 millones de interacciones agregadas "
                "en más de 150 países para entender la colaboración con IA "
                "en el trabajo."
            ),
            source_url="https://example.com/atlas",
            territory="ai_innovation_future",
        )
        experiment = build_audience_experiment(atlas)
        self.assertEqual(
            experiment["learning_question"],
            "¿La IA ya te ahorra tiempo o todavía te complica el trabajo?",
        )
        self.assertEqual(
            experiment["answer_options"],
            ["Me ahorra tiempo", "Me complica", "Todavía no la uso"],
        )


if __name__ == "__main__":
    unittest.main()
