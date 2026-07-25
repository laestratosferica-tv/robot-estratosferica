import json
import tempfile
import unittest
from pathlib import Path

from media_factory.config import ConfigurationError, load_config, validate_config
from media_factory.editor import evaluate_candidate
from media_factory.models import Candidate
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
        decision = evaluate_candidate(
            Candidate(
                title="Historia",
                source_url="https://example.com/story",
                territory="ai_innovation_future",
            ),
            self.config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_queue([decision], Path(directory) / "queue.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "dry_run")
        self.assertFalse(payload["publishing_enabled"])

    def test_unsafe_configuration_is_blocked(self) -> None:
        unsafe = json.loads(json.dumps(self.config))
        unsafe["safe_mode"]["publishing_enabled"] = True
        with self.assertRaises(ConfigurationError):
            validate_config(unsafe)


if __name__ == "__main__":
    unittest.main()
