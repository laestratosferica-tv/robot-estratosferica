import unittest

from media_factory.editorial_quality import (
    substantive_summary_issue,
    text_is_equivalent,
)


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


if __name__ == "__main__":
    unittest.main()
