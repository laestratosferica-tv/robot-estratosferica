import json
import tempfile
import unittest
from pathlib import Path

from media_factory.commercial import detect_opportunity
from media_factory.config import ConfigurationError, load_config, validate_config
from media_factory.editor import evaluate_candidate
from media_factory.metrics import build_measurement_plan
from media_factory.models import Candidate, PipelineItem
from media_factory.queue import save_queue


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "editorial_v1.json"


class EditorialV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_configuration_is_safe(self) -> None:
        safe = self.config["safe_mode"]
        self.assertTrue(safe["dry_run"])
        self.assertFalse(safe["publishing_enabled"])
        self.assertFalse(safe["social_tokens_allowed"])

    def test_good_candidate_reaches_review(self) -> None:
        candidate = Candidate(
            title="La final regional cambia la economía de los esports",
            source_url="https://example.com/source",
            territory="gaming_esports",
            signals={key: 0.9 for key in self.config["editorial_score"]["weights"]},
        )
        decision = evaluate_candidate(candidate, self.config)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.state, "needs_review")
        self.assertGreaterEqual(decision.score, 65)

    def test_unverified_candidate_is_rejected(self) -> None:
        candidate = Candidate(
            title="Rumor sin confirmar",
            source_url="https://example.com/rumor",
            territory="gaming_esports",
            is_verified=False,
            signals={key: 1 for key in self.config["editorial_score"]["weights"]},
        )
        decision = evaluate_candidate(candidate, self.config)
        self.assertFalse(decision.accepted)
        self.assertIn("unverified_rumor", decision.rejection_reasons)

    def test_queue_cannot_publish(self) -> None:
        candidate = Candidate(
            title="Historia",
            source_url="https://example.com/story",
            territory="ai_innovation_future",
        )
        decision = evaluate_candidate(candidate, self.config)
        opportunity = detect_opportunity(candidate, decision)
        item = PipelineItem(
            decision=decision,
            commercial_opportunity=opportunity,
            measurement_plan=build_measurement_plan(decision, opportunity),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_queue([item], Path(directory) / "queue.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "dry_run")
        self.assertFalse(payload["publishing_enabled"])
        self.assertFalse(payload["external_actions_enabled"])

    def test_commercial_signal_remains_research_only(self) -> None:
        candidate = Candidate(
            title="Nueva activación digital regional",
            source_url="https://example.com/activation",
            territory="brands_activations",
            signals={key: 1 for key in self.config["editorial_score"]["weights"]},
        )
        decision = evaluate_candidate(candidate, self.config)
        opportunity = detect_opportunity(candidate, decision)
        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.status, "research_only")
        self.assertIn("sin aprobación humana", opportunity.next_step)

    def test_unsafe_configuration_is_blocked(self) -> None:
        unsafe = json.loads(json.dumps(self.config))
        unsafe["safe_mode"]["publishing_enabled"] = True
        with self.assertRaises(ConfigurationError):
            validate_config(unsafe)


if __name__ == "__main__":
    unittest.main()
