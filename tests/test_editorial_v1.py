import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from media_factory.commercial import detect_opportunity
from media_factory.cli import run_factory
from media_factory.config import ConfigurationError, load_config, validate_config
from media_factory.editor import evaluate_candidate
from media_factory.guardrails import validate_content_package, validate_storyboard
from media_factory.metrics import build_measurement_plan
from media_factory.models import Candidate, PipelineItem
from media_factory.queue import save_queue
from media_factory.radar import RadarRejected, load_source_registry, normalize_story
from media_factory.selector import build_selection_report, rank_opportunities
from media_factory.studio import build_content_package
from media_factory.storyboard import build_storyboard
from media_factory.strategy import classify_candidate, load_content_strategy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "editorial_v1.json"
SOURCES_PATH = ROOT / "config" / "sources_v1.json"
STRATEGY_PATH = ROOT / "config" / "content_strategy_v1.json"


class EditorialV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.sources = load_source_registry(SOURCES_PATH)
        self.strategy = load_content_strategy(STRATEGY_PATH)

    def _classified(self, candidate: Candidate) -> Candidate:
        return replace(
            candidate,
            strategic_classification=classify_candidate(
                candidate,
                self.strategy,
            ),
        )

    def _selected(self, candidate: Candidate, decision):
        selections = rank_opportunities(
            [candidate],
            [decision],
            self.config,
        )
        self.assertTrue(selections[0].selected)
        return selections[0], build_selection_report(selections)

    def test_configuration_is_safe(self) -> None:
        safe = self.config["safe_mode"]
        self.assertTrue(safe["dry_run"])
        self.assertFalse(safe["publishing_enabled"])
        self.assertFalse(safe["social_tokens_allowed"])

    def test_good_candidate_reaches_review(self) -> None:
        candidate = Candidate(
            title="La final regional cambia la economía de los esports",
            summary=(
                "El torneo aumentó la bolsa de premios y sumó una nueva "
                "clasificación para equipos latinoamericanos."
            ),
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
        candidate = self._classified(
            Candidate(
                title="Historia",
                summary=(
                    "La fuente documenta una función que reduce el tiempo "
                    "necesario para completar una tarea creativa."
                ),
                source_url="https://example.com/story",
                territory="ai_innovation_future",
                signals={
                    key: 1
                    for key in self.config["editorial_score"]["weights"]
                },
            )
        )
        decision = evaluate_candidate(candidate, self.config)
        selection, selection_report = self._selected(candidate, decision)
        opportunity = detect_opportunity(candidate, decision)
        item = PipelineItem(
            candidate=candidate,
            decision=decision,
            opportunity_selection=selection,
            commercial_opportunity=opportunity,
            content_package=build_content_package(
                candidate, decision, opportunity
            ),
            storyboard=None,
            measurement_plan=build_measurement_plan(decision, opportunity),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_queue(
                [item],
                Path(directory) / "queue.json",
                selection_report=selection_report,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "dry_run")
        self.assertFalse(payload["publishing_enabled"])
        self.assertFalse(payload["external_actions_enabled"])
        self.assertEqual(payload["schema_version"], "review_queue_v1")
        self.assertTrue(payload["human_approval_required"])
        review = payload["items"][0]["review"]
        self.assertEqual(review["status"], "pending_human_approval")
        self.assertTrue(review["requires_human_approval"])
        self.assertFalse(review["approved"])
        self.assertFalse(review["publish_allowed"])
        self.assertEqual(
            review["source"]["url"],
            "https://example.com/story",
        )
        self.assertTrue(review["candidate_id"])
        self.assertTrue(review["content_fingerprint"])
        self.assertTrue(review["anti_duplicate_id"])
        strategy = review["strategy"]
        self.assertEqual(
            strategy["content_product_id"],
            "esto_cambia_el_juego",
        )
        self.assertTrue(strategy["audience_hypothesis"])
        self.assertTrue(strategy["expected_community_action"])
        self.assertTrue(strategy["primary_metric"])
        self.assertTrue(strategy["commercial_path"])
        self.assertFalse(strategy["publishing_enabled"])
        self.assertTrue(review["opportunity_selection"]["selected"])
        editorial_test = review["editorial_test"]
        self.assertEqual(editorial_test["state"], "draft")
        self.assertTrue(editorial_test["objective"])
        self.assertTrue(editorial_test["expected_interaction"])
        self.assertTrue(editorial_test["interaction_prompt"])
        self.assertTrue(editorial_test["primary_metric"])
        self.assertFalse(editorial_test["views_only_success_allowed"])
        self.assertFalse(editorial_test["publishing_enabled"])

    def test_review_ids_are_stable_for_the_same_radar_candidate(self) -> None:
        candidate = self._classified(
            Candidate(
                candidate_id="radar-candidate-1",
                title="Historia estable",
                summary=(
                    "La fuente confirma un cambio de formato para la próxima "
                    "temporada competitiva regional."
                ),
                source_url="https://example.com/stable",
                source_id="source",
                territory="gaming_esports",
                signals={
                    key: 1
                    for key in self.config["editorial_score"]["weights"]
                },
            )
        )
        decision = evaluate_candidate(candidate, self.config)
        selection, selection_report = self._selected(candidate, decision)
        opportunity = detect_opportunity(candidate, decision)
        item = PipelineItem(
            candidate=candidate,
            decision=decision,
            opportunity_selection=selection,
            commercial_opportunity=opportunity,
            content_package=build_content_package(
                candidate, decision, opportunity
            ),
            storyboard=None,
            measurement_plan=build_measurement_plan(decision, opportunity),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = json.loads(
                save_queue(
                    [item],
                    root / "first.json",
                    selection_report=selection_report,
                ).read_text(
                    encoding="utf-8"
                )
            )
            second = json.loads(
                save_queue(
                    [item],
                    root / "second.json",
                    selection_report=selection_report,
                ).read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(
            first["items"][0]["review"],
            second["items"][0]["review"],
        )
        self.assertEqual(
            first["items"][0]["review"]["candidate_id"],
            "radar-candidate-1",
        )

    def test_commercial_signal_remains_research_only(self) -> None:
        candidate = Candidate(
            title="Nueva activación digital regional",
            summary=(
                "La activación permite probar el producto dentro de una "
                "experiencia interactiva en tres ciudades."
            ),
            source_url="https://example.com/activation",
            territory="brands_activations",
            signals={key: 1 for key in self.config["editorial_score"]["weights"]},
        )
        decision = evaluate_candidate(candidate, self.config)
        opportunity = detect_opportunity(candidate, decision)
        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.status, "research_only")
        self.assertIn("sin aprobación humana", opportunity.next_step)

    def test_halo_game_pass_candidate_is_rejected_and_next_reaches_queue(self):
        halo_title = (
            "Próximamente en XBOX Game Pass: Halo: Campaign Evolved, "
            "Beast of Reincarnation y más"
        )
        weights = self.config["editorial_score"]["weights"]
        raw_candidates = [
            {
                "candidate_id": "halo-game-pass",
                "title": halo_title,
                "summary": halo_title,
                "source_url": "https://news.xbox.com/es-latam/halo-game-pass/",
                "source_id": "xbox_wire_es_latam",
                "territory": "gaming_esports",
                "signals": {key: 1 for key in weights},
            },
            {
                "candidate_id": "next-substantive-story",
                "title": "Xbox amplía sus controles de accesibilidad",
                "summary": (
                    "La actualización incorpora subtítulos configurables y "
                    "nuevos controles de contraste para jugadores."
                ),
                "source_url": "https://news.xbox.com/es-latam/accessibility/",
                "source_id": "xbox_wire_es_latam",
                "territory": "gaming_esports",
                "signals": {key: 0.9 for key in weights},
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "candidates.json"
            queue_path = root / "queue.json"
            input_path.write_text(
                json.dumps(raw_candidates, ensure_ascii=False),
                encoding="utf-8",
            )
            result = run_factory(
                CONFIG_PATH,
                input_path,
                queue_path,
                talent_config_path=ROOT / "config" / "talent_v1.json",
            )
            queue = json.loads(queue_path.read_text(encoding="utf-8"))

        self.assertEqual(result["rejected_count"], 1)
        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(len(queue["items"]), 1)
        self.assertEqual(
            queue["items"][0]["review"]["candidate_id"],
            "next-substantive-story",
        )
        ranked = queue["opportunity_selection"]["ranked_candidates"]
        halo = next(
            item for item in ranked
            if item["candidate_id"] == "halo-game-pass"
        )
        self.assertFalse(halo["eligible"])
        self.assertFalse(halo["selected"])
        self.assertIn(
            "summary_equivalent_to_title",
            halo["blocking_reasons"],
        )
        self.assertNotIn(
            halo_title,
            str(queue["items"]),
        )

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
        self.assertTrue(package.content_punch["gate_passed"])
        self.assertEqual(
            package.content_punch["evidence_origin"],
            "candidate.summary",
        )
        self.assertIn(
            package.content_punch["tension_question"],
            package.platform_copy["instagram"],
        )
        self.assertNotIn("?.", package.platform_copy["threads"])
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

    def test_threads_copy_does_not_add_a_period_after_a_question(self) -> None:
        candidate = Candidate(
            title="Google presenta AI & Economy ATLAS",
            summary=(
                "El estudio analiza 15 millones de interacciones agregadas "
                "en más de 150 países para entender la IA en el trabajo."
            ),
            source_url="https://example.com/atlas",
            territory="ai_innovation_future",
            signals={key: 1 for key in self.config["editorial_score"]["weights"]},
        )
        decision = evaluate_candidate(candidate, self.config)
        package = build_content_package(candidate, decision, None)
        self.assertIsNotNone(package)
        self.assertNotIn("?.", package.platform_copy["threads"])
        self.assertTrue(
            package.platform_copy["threads"].startswith(
                "¿LA IA TE POTENCIA O TE REEMPLAZA? "
            )
        )

    def test_threads_copy_fits_long_verified_feed_summaries(self) -> None:
        candidate = Candidate(
            title=(
                "Play More Ubisoft Games on PC: New Content Now Available "
                "for Xbox on PC"
            ),
            summary=" ".join(
                [
                    "The official source explains the current catalog,",
                    "availability, devices, conditions and regional rollout.",
                ]
                * 12
            ),
            source_url="https://news.xbox.com/en-us/example",
            territory="gaming_esports",
            signals={
                key: 1 for key in self.config["editorial_score"]["weights"]
            },
        )
        decision = evaluate_candidate(candidate, self.config)
        package = build_content_package(candidate, decision, None)
        self.assertIsNotNone(package)
        threads_copy = package.platform_copy["threads"]
        self.assertLessEqual(len(threads_copy), 500)
        self.assertTrue(
            threads_copy.endswith(
                package.content_punch["tension_question"]
            )
        )
        self.assertEqual(validate_content_package(package), [])

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
