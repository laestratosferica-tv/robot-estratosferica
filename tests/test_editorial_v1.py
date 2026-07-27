import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from media_factory.commercial import detect_opportunity
from media_factory.config import ConfigurationError, load_config, validate_config
from media_factory.editor import evaluate_candidate
from media_factory.guardrails import validate_content_package, validate_storyboard
from media_factory.metrics import build_measurement_plan
from media_factory.models import Candidate, PipelineItem
from media_factory.queue import save_queue
from media_factory.radar import RadarRejected, load_source_registry, normalize_story
from media_factory.studio import build_content_package
from media_factory.storyboard import build_storyboard


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "editorial_v1.json"
SOURCES_PATH = ROOT / "config" / "sources_v1.json"


class EditorialV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.sources = load_source_registry(SOURCES_PATH)

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
            candidate=candidate,
            decision=decision,
            commercial_opportunity=opportunity,
            content_package=build_content_package(
                candidate, decision, opportunity
            ),
            storyboard=None,
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

    def test_radar_accepts_recent_allowed_source(self) -> None:
        candidate = normalize_story(
            {
                "title": "XBOX y Meta amplían una experiencia de juego",
                "source_url": "https://news.xbox.com/es-latam/example",
                "source_id": "xbox_wire_es_latam",
                "published_at": "2026-07-21",
                "territory": "brands_activations",
            },
            self.sources,
            today=date(2026, 7, 25),
        )
        self.assertEqual(candidate.source_id, "xbox_wire_es_latam")

    def test_radar_blocks_betting_content(self) -> None:
        with self.assertRaisesRegex(RadarRejected, "blocked_topic"):
            normalize_story(
                {
                    "title": "New esports betting market",
                    "source_url": "https://esportsinsider.com/example",
                    "source_id": "esports_insider",
                    "published_at": "2026-07-24",
                    "territory": "gaming_esports",
                },
                self.sources,
                today=date(2026, 7, 25),
            )

    def test_studio_builds_reviewable_multiplatform_draft(self) -> None:
        candidate = Candidate(
            title="XBOX y Meta amplían una experiencia de juego",
            summary="Una alianza integra dos servicios de juego.",
            source_url="https://news.xbox.com/es-latam/example",
            territory="brands_activations",
            signals={key: 1 for key in self.config["editorial_score"]["weights"]},
        )
        decision = evaluate_candidate(candidate, self.config)
        opportunity = detect_opportunity(candidate, decision)
        package = build_content_package(candidate, decision, opportunity)
        self.assertIsNotNone(package)
        self.assertEqual(package.state, "draft")
        self.assertEqual(package.format_id, "brand_play")
        self.assertEqual(
            set(package.platform_copy),
            {"instagram", "facebook", "youtube", "threads"},
        )
        self.assertTrue(package.audience_experiment["experiment_id"])
        self.assertFalse(
            package.audience_experiment["publishing_enabled"]
        )
        self.assertNotIn("tiktok", package.platform_copy)
        self.assertEqual(validate_content_package(package), [])

    def test_studio_does_not_package_rejected_story(self) -> None:
        candidate = Candidate(
            title="Rumor",
            source_url="",
            territory="gaming_esports",
        )
        decision = evaluate_candidate(candidate, self.config)
        self.assertIsNone(build_content_package(candidate, decision, None))

    def test_storyboard_is_safe_30_second_production_plan(self) -> None:
        candidate = Candidate(
            title="XBOX y Meta amplían una experiencia de juego",
            summary="Una alianza integra dos servicios de juego.",
            source_url="https://news.xbox.com/es-latam/example",
            territory="brands_activations",
            signals={key: 1 for key in self.config["editorial_score"]["weights"]},
        )
        decision = evaluate_candidate(candidate, self.config)
        opportunity = detect_opportunity(candidate, decision)
        package = build_content_package(candidate, decision, opportunity)
        storyboard = build_storyboard(candidate, package)
        self.assertIsNotNone(storyboard)
        self.assertEqual(storyboard.duration_seconds, 30)
        self.assertEqual(len(storyboard.scenes), 6)
        self.assertFalse(storyboard.production_enabled)
        self.assertTrue(storyboard.captions_required)
        self.assertEqual(validate_storyboard(storyboard), [])
        self.assertEqual(
            storyboard.scenes[-1].end_second,
            storyboard.duration_seconds,
        )

    def test_unsafe_configuration_is_blocked(self) -> None:
        unsafe = json.loads(json.dumps(self.config))
        unsafe["safe_mode"]["publishing_enabled"] = True
        with self.assertRaises(ConfigurationError):
            validate_config(unsafe)


if __name__ == "__main__":
    unittest.main()
