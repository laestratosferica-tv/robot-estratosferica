import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from review_panel import record_decision, render_panel, validate_queue


def safe_queue() -> dict:
    return {
        "schema_version": "review_queue_v1",
        "mode": "dry_run",
        "publishing_enabled": False,
        "external_actions_enabled": False,
        "human_approval_required": True,
        "items": [
            {
                "story": {"title": "Final regional de esports"},
                "content_package": {
                    "factual_summary": "Una final con impacto regional."
                },
                "review": {
                    "review_id": "review-demo",
                    "candidate_id": "candidate-demo",
                    "content_fingerprint": "fingerprint-demo",
                    "status": "pending_human_approval",
                    "requires_human_approval": True,
                    "approved": False,
                    "publish_allowed": False,
                    "source": {"url": "https://example.com/source"},
                    "final_text_by_platform": {
                        "threads": "¿Quién gana la final?"
                    },
                    "strategy": {
                        "content_product_id": "encuesta_prediccion",
                    },
                    "opportunity_selection": {
                        "rank": 1,
                        "score": 88,
                        "rationale": ["alta conversación"],
                    },
                    "editorial_test": {
                        "objective": "Construir comunidad",
                        "expected_interaction": "Votos",
                        "interaction_prompt": "¿Quién gana?",
                        "answer_options": ["Equipo A", "Equipo B"],
                        "primary_metric": "votos_validos",
                    },
                },
            }
        ],
    }


class ReviewPanelTests(unittest.TestCase):
    def test_panel_explains_approval_does_not_publish(self) -> None:
        page = render_panel(safe_queue())
        self.assertIn("aprobar aquí nunca publica", page)
        self.assertIn("Final regional de esports", page)
        self.assertIn("votos_validos", page)
        self.assertIn("Abrir fuente original", page)

    def test_editorial_approval_is_recorded_without_publish_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            entry = record_decision(
                safe_queue(),
                path,
                review_id="review-demo",
                decision="approved_editorially",
                reviewer="José Luis",
                now=datetime(2026, 7, 28, tzinfo=timezone.utc),
            )
            ledger = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(entry["decision"], "approved_editorially")
        self.assertTrue(entry["editorial_approval_only"])
        self.assertFalse(entry["publish_allowed"])
        self.assertFalse(ledger["publishing_enabled"])
        self.assertFalse(ledger["external_actions_enabled"])

    def test_rejection_requires_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requiere un motivo"):
                record_decision(
                    safe_queue(),
                    Path(directory) / "decisions.json",
                    review_id="review-demo",
                    decision="rejected",
                )

    def test_same_review_cannot_be_decided_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            record_decision(
                safe_queue(),
                path,
                review_id="review-demo",
                decision="approved_editorially",
            )
            with self.assertRaisesRegex(ValueError, "ya tiene"):
                record_decision(
                    safe_queue(),
                    path,
                    review_id="review-demo",
                    decision="approved_editorially",
                )

    def test_unsafe_queue_is_blocked(self) -> None:
        queue = safe_queue()
        queue["publishing_enabled"] = True
        with self.assertRaisesRegex(ValueError, "publishing_enabled"):
            validate_queue(queue)

    def test_unknown_review_id_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "desconocido"):
                record_decision(
                    safe_queue(),
                    Path(directory) / "decisions.json",
                    review_id="review-other",
                    decision="approved_editorially",
                )


if __name__ == "__main__":
    unittest.main()
