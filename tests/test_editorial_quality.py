import json
import unittest
from pathlib import Path

from media_factory.editorial_quality import (
    substantive_summary_issue,
    text_is_equivalent,
)

ROOT = Path(__file__).resolve().parents[1]


class EditorialQualityTests(unittest.TestCase):
    def test_empty_summary_is_not_substantive(self):
        self.assertEqual(
            substantive_summary_issue("Un anuncio confirmado", "   "),
            "missing_substantive_summary",
        )

    def test_case_accents_and_punctuation_do_not_hide_title_duplication(self):
        self.assertTrue(text_is_equivalent(
            "Próximamente: edición y evolución de Halo",
            "PROXIMAMENTE, edicion y evolucion de Halo.",
        ))

    def test_near_duplicate_summary_is_blocked_deterministically(self):
        title = (
            "Xbox anuncia nuevos juegos de Game Pass para agosto"
        )
        summary = (
            "Xbox anuncia oficialmente nuevos juegos de Game Pass para agosto"
        )
        self.assertEqual(
            substantive_summary_issue(title, summary),
            "summary_equivalent_to_title",
        )

    def test_distinct_verified_fact_is_allowed(self):
        self.assertIsNone(substantive_summary_issue(
            "Xbox anuncia nuevos juegos de Game Pass para agosto",
            (
                "La primera tanda estará disponible el 4 de agosto e incluye "
                "dos estrenos desde su día de lanzamiento."
            ),
        ))

    def test_all_real_run_30399824136_visual_summaries_are_rejected(self):
        fixture = json.loads(
            (
                ROOT
                / "fixtures"
                / "real_visual_summaries_run_30399824136.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(fixture), 5)
        for item in fixture:
            with self.subTest(title=item["title"]):
                self.assertEqual(
                    substantive_summary_issue(
                        item["title"],
                        item["summary"],
                    ),
                    item["expected_rejection"],
                )

    def test_visual_reference_with_a_verifiable_fact_is_allowed(self):
        self.assertIsNone(substantive_summary_issue(
            "Satélite registra cambios en un glaciar",
            (
                "La imagen satelital confirma que el glaciar perdió 12 % "
                "de su superficie desde 2020."
            ),
        ))

    def test_truncated_feed_excerpt_is_rejected(self):
        for summary in (
            "El encuentro hizo historia y se c…",
            "Google Earth documenta el estadio...",
            "La plataforma incorpora una experiencia…",
        ):
            with self.subTest(summary=summary):
                self.assertEqual(
                    substantive_summary_issue("Historia confirmada", summary),
                    "summary_truncated",
                )

    def test_complete_summary_with_period_is_not_truncated(self):
        self.assertIsNone(substantive_summary_issue(
            "Google Earth recrea un estadio",
            (
                "La plataforma incorpora un recorrido que documenta momentos "
                "históricos del Estadio Azteca."
            ),
        ))

    def test_short_labels_are_not_mistaken_for_reported_facts(self):
        for summary in (
            "AI launch graphic",
            "Football documentary",
            "New product cover",
        ):
            with self.subTest(summary=summary):
                self.assertIsNotNone(
                    substantive_summary_issue("Titular distinto", summary)
                )

    def test_factual_summaries_pass_across_multiple_topics(self):
        cases = (
            (
                "Nuevo modelo de IA",
                "La empresa confirmó que el modelo estará disponible el 3 de agosto.",
            ),
            (
                "Cambios en una liga regional",
                "El torneo sumó cuatro equipos y aumentó la bolsa de premios.",
            ),
            (
                "Análisis del empleo",
                "El informe analiza 15 millones de registros de 150 países.",
            ),
            (
                "Nueva función de acceso",
                "La actualización incorpora verificación facial opcional para iniciar sesión.",
            ),
        )
        for title, summary in cases:
            with self.subTest(title=title):
                self.assertIsNone(
                    substantive_summary_issue(title, summary)
                )


if __name__ == "__main__":
    unittest.main()
