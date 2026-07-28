import json
import tempfile
import unittest
from pathlib import Path

from media_factory.models import Candidate
from media_factory.strategy import (
    StrategyConfigurationError,
    classify_candidate,
    load_content_strategy,
    validate_content_strategy,
    validate_strategy_decision,
)


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "config" / "content_strategy_v1.json"


class StrategyClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.strategy = load_content_strategy(STRATEGY_PATH)

    def test_verified_gaming_news_becomes_radar_product(self) -> None:
        decision = classify_candidate(
            Candidate(
                title="Xbox amplía Game Pass en Latinoamérica",
                summary="La plataforma anunció nuevos accesos.",
                source_url="https://news.xbox.com/es-latam/example",
                source_id="xbox_wire_es_latam",
                territory="gaming_esports",
            ),
            self.strategy,
        )

        self.assertEqual(
            decision["content_product_id"],
            "radar_estratosferico",
        )
        self.assertEqual(decision["funnel_stage"], "attract")
        self.assertEqual(decision["rights_state"], "original_owned")
        self.assertEqual(validate_strategy_decision(decision), [])

    def test_ai_story_with_practical_impact_becomes_explainer(self) -> None:
        decision = classify_candidate(
            Candidate(
                title="Google explica el impacto de la IA en el trabajo",
                summary="La herramienta cambia tareas de desarrolladores.",
                source_url="https://blog.google/example",
                source_id="google_blog",
                territory="ai_innovation_future",
            ),
            self.strategy,
        )

        self.assertEqual(
            decision["content_product_id"],
            "esto_cambia_el_juego",
        )
        self.assertEqual(decision["funnel_stage"], "learn")
        self.assertEqual(
            decision["expected_community_action"],
            "guardar_compartir_o_preguntar",
        )

    def test_epic_play_stays_link_only_and_never_enables_actions(self) -> None:
        decision = classify_candidate(
            {
                "title": "ACE imposible en la final de VALORANT",
                "description": "La jugada definió la partida.",
                "editorial_lane": "epic_plays_and_creators",
                "rights": {"state": "link_only_unverified"},
            },
            self.strategy,
        )

        self.assertEqual(
            decision["content_product_id"],
            "jugada_estratosferica",
        )
        self.assertEqual(
            decision["rights_state"],
            "official_embed_or_link",
        )
        self.assertFalse(decision["publishing_enabled"])
        self.assertFalse(decision["broadcasting_enabled"])
        self.assertFalse(decision["external_actions_enabled"])
        self.assertTrue(decision["requires_human_review"])

    def test_setup_requires_technology_and_purchase_intent(self) -> None:
        decision = classify_candidate(
            {
                "title": "¿Qué mouse gamer vale la pena comprar?",
                "summary": "Comparativa de precio y desempeño.",
                "territory": "gaming_esports",
            },
            self.strategy,
        )

        self.assertEqual(decision["content_product_id"], "setup_real")
        self.assertEqual(decision["funnel_stage"], "monetize")
        self.assertEqual(decision["primary_metric"], "clics_atribuidos")

    def test_lifestyle_requires_an_explicit_gamer_connection(self) -> None:
        unrelated = classify_candidate(
            {
                "title": "Nuevo menú de café en Bogotá",
                "summary": "Una experiencia gastronómica local.",
                "territory": "sport_technology_entertainment",
            },
            self.strategy,
        )
        connected = classify_candidate(
            {
                "title": "Café para una noche de esports",
                "summary": "La experiencia conecta gastronomía y gamers.",
                "territory": "sport_technology_entertainment",
            },
            self.strategy,
        )

        self.assertEqual(
            unrelated["content_product_id"],
            "radar_estratosferico",
        )
        self.assertEqual(
            connected["content_product_id"],
            "cultura_en_modo_gamer",
        )

    def test_unverified_broadcast_is_classified_but_blocked(self) -> None:
        decision = classify_candidate(
            {
                "title": "Final regional disponible para retransmisión",
                "content_type": "broadcast_opportunity",
            },
            self.strategy,
        )

        self.assertEqual(
            decision["content_product_id"],
            "arena_estratosferica",
        )
        self.assertEqual(decision["rights_state"], "unverified_blocked")
        self.assertFalse(decision["rights_ready_for_draft"])
        self.assertFalse(decision["broadcasting_enabled"])

    def test_unsafe_strategy_configuration_is_rejected(self) -> None:
        unsafe = json.loads(json.dumps(self.strategy))
        unsafe["safety"]["publishing_enabled"] = True

        with self.assertRaises(StrategyConfigurationError):
            validate_content_strategy(unsafe)

    def test_decision_is_stable_for_the_same_candidate(self) -> None:
        candidate = {
            "title": "Ranking de mapas preferidos",
            "content_type": "poll",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            first = classify_candidate(candidate, self.strategy)
            path.write_text(json.dumps(first, sort_keys=True), encoding="utf-8")
            second = classify_candidate(candidate, self.strategy)

        self.assertEqual(first, second)
        self.assertEqual(
            first["content_product_id"],
            "comunidad_decide",
        )


if __name__ == "__main__":
    unittest.main()
