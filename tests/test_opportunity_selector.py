import json
import unittest
from dataclasses import replace
from pathlib import Path

from media_factory.config import (
    ConfigurationError,
    load_config,
    validate_config,
)
from media_factory.editor import evaluate_candidate
from media_factory.models import Candidate
from media_factory.selector import build_selection_report, rank_opportunities
from media_factory.strategy import classify_candidate, load_content_strategy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "editorial_v1.json"
STRATEGY_PATH = ROOT / "config" / "content_strategy_v1.json"


class OpportunitySelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.strategy = load_content_strategy(STRATEGY_PATH)

    def _candidate(
        self,
        *,
        candidate_id: str,
        title: str,
        signals: dict[str, float],
        strategy_override: dict | None = None,
    ) -> Candidate:
        candidate = Candidate(
            candidate_id=candidate_id,
            title=title,
            summary="Historia verificada con una consecuencia concreta.",
            source_url=f"https://example.com/{candidate_id}",
            source_id="source",
            territory="gaming_esports",
            signals=signals,
        )
        strategy = classify_candidate(candidate, self.strategy)
        if strategy_override:
            strategy.update(strategy_override)
        return replace(
            candidate,
            strategic_classification=strategy,
        )

    def test_selects_one_candidate_for_community_and_business_value(self):
        viral_but_empty = self._candidate(
            candidate_id="viral",
            title="Clip con muchas vistas",
            signals={
                "latam_relevance": 0.7,
                "explanatory_value": 0.7,
                "angle_originality": 0.7,
                "verifiability": 1,
                "conversation_potential": 0.7,
                "commercial_potential": 0.4,
            },
        )
        useful = self._candidate(
            candidate_id="useful",
            title="La comunidad compara el cambio más importante",
            signals={
                "latam_relevance": 0.9,
                "explanatory_value": 0.95,
                "angle_originality": 0.9,
                "verifiability": 1,
                "conversation_potential": 0.95,
                "commercial_potential": 0.9,
            },
        )
        candidates = [viral_but_empty, useful]
        decisions = [
            evaluate_candidate(candidate, self.config)
            for candidate in candidates
        ]

        selections = rank_opportunities(
            candidates,
            decisions,
            self.config,
        )
        report = build_selection_report(selections)

        self.assertEqual(report["selected_count"], 1)
        self.assertEqual(report["selected_candidate_id"], "useful")
        chosen = next(item for item in selections if item.selected)
        self.assertEqual(chosen.candidate_id, "useful")
        self.assertEqual(chosen.candidate_title, useful.title)
        self.assertEqual(
            chosen.content_product_id,
            "radar_estratosferico",
        )
        self.assertTrue(chosen.objective)
        self.assertTrue(chosen.expected_interaction)
        self.assertTrue(chosen.primary_metric)
        self.assertFalse(chosen.views_only_success_allowed)

    def test_rights_blocked_candidate_cannot_be_selected(self):
        blocked = self._candidate(
            candidate_id="blocked",
            title="Final sin autorización",
            signals={
                key: 1
                for key in self.config["editorial_score"]["weights"]
            },
            strategy_override={
                "rights_state": "unverified_blocked",
                "rights_ready_for_draft": False,
            },
        )
        decision = evaluate_candidate(blocked, self.config)

        selection = rank_opportunities(
            [blocked],
            [decision],
            self.config,
        )[0]

        self.assertFalse(selection.eligible)
        self.assertFalse(selection.selected)
        self.assertIn(
            "rights_not_ready_for_draft",
            selection.blocking_reasons,
        )

    def test_ranking_is_stable_when_scores_tie(self):
        signals = {
            key: 0.9
            for key in self.config["editorial_score"]["weights"]
        }
        candidates = [
            self._candidate(
                candidate_id="candidate-b",
                title="Historia B",
                signals=signals,
            ),
            self._candidate(
                candidate_id="candidate-a",
                title="Historia A",
                signals=signals,
            ),
        ]
        decisions = [
            evaluate_candidate(candidate, self.config)
            for candidate in candidates
        ]

        first = rank_opportunities(candidates, decisions, self.config)
        second = rank_opportunities(candidates, decisions, self.config)

        self.assertEqual(first, second)
        selected = next(item for item in first if item.selected)
        self.assertEqual(selected.candidate_id, "candidate-a")

    def test_unsafe_selector_configuration_is_rejected(self):
        unsafe = json.loads(json.dumps(self.config))
        unsafe["opportunity_selector"]["views_only_success_allowed"] = True

        with self.assertRaises(ConfigurationError):
            validate_config(unsafe)


if __name__ == "__main__":
    unittest.main()
