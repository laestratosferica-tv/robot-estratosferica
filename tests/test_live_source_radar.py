import json
import tempfile
import time
import unittest
from unittest.mock import patch
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

from live_source_radar import (
    LiveRadarError,
    _ApprovedRedirectHandler,
    collect_live_candidates,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "config" / "sources_v1.json"


def _parsed_date(value: str) -> time.struct_time:
    return time.strptime(value, "%Y-%m-%d")


class LiveSourceRadarTests(unittest.TestCase):
    def test_article_redirect_to_unapproved_domain_is_blocked(self):
        handler = _ApprovedRedirectHandler(["blog.google"])
        with self.assertRaisesRegex(
            LiveRadarError,
            "article_redirect_domain_mismatch",
        ):
            handler.redirect_request(
                None,
                None,
                302,
                "Found",
                {},
                "https://example.com/unapproved/",
            )

    def _collect_with(self, parser):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=root / "candidates.json",
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
        return report

    def test_all_sources_inaccessible_is_unhealthy(self):
        def parser(_feed_url: str):
            raise URLError("network unavailable")

        report = self._collect_with(parser)

        self.assertFalse(report["healthy"])
        self.assertEqual(report["status"], "network_failure")
        self.assertEqual(
            [source["status"] for source in report["sources"]],
            ["inaccessible", "inaccessible"],
        )
        with patch(
            "live_source_radar.collect_live_candidates",
            return_value=report,
        ), patch("sys.argv", ["live_source_radar.py"]):
            self.assertEqual(main(), 1)

    def test_accessible_empty_sources_are_healthy(self):
        report = self._collect_with(
            lambda _feed_url: SimpleNamespace(entries=[])
        )

        self.assertTrue(report["healthy"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(
            [source["status"] for source in report["sources"]],
            ["accessible_empty", "accessible_empty"],
        )

    def test_accessible_and_failed_sources_report_partial_health(self):
        def parser(feed_url: str):
            if "xbox" in feed_url:
                return SimpleNamespace(entries=[])
            raise ValueError("malformed feed")

        report = self._collect_with(parser)

        self.assertTrue(report["healthy"])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(
            [source["status"] for source in report["sources"]],
            ["accessible_empty", "error"],
        )

    def test_all_parse_errors_are_unhealthy_source_failure(self):
        report = self._collect_with(
            lambda _feed_url: SimpleNamespace(
                entries=[],
                bozo=True,
                bozo_exception=ValueError("malformed XML"),
            )
        )

        self.assertFalse(report["healthy"])
        self.assertEqual(report["status"], "source_failure")
        self.assertEqual(
            [source["status"] for source in report["sources"]],
            ["error", "error"],
        )
        with patch(
            "live_source_radar.collect_live_candidates",
            return_value=report,
        ), patch("sys.argv", ["live_source_radar.py"]):
            self.assertEqual(main(), 1)

    def test_network_and_parse_error_without_accessible_source_is_unhealthy(self):
        def parser(feed_url: str):
            if "xbox" in feed_url:
                raise URLError("network unavailable")
            raise ValueError("malformed feed")

        report = self._collect_with(parser)

        self.assertFalse(report["healthy"])
        self.assertEqual(report["status"], "source_failure")
        self.assertNotEqual(report["status"], "partial")
        self.assertEqual(
            [source["status"] for source in report["sources"]],
            ["inaccessible", "error"],
        )

    def test_partial_requires_an_accessible_source(self):
        def no_accessible_source(feed_url: str):
            if "xbox" in feed_url:
                raise URLError("network unavailable")
            raise ValueError("malformed feed")

        def one_accessible_source(feed_url: str):
            if "xbox" in feed_url:
                return SimpleNamespace(entries=[])
            raise ValueError("malformed feed")

        failures = self._collect_with(no_accessible_source)
        partial = self._collect_with(one_accessible_source)

        self.assertNotEqual(failures["status"], "partial")
        self.assertFalse(failures["healthy"])
        self.assertEqual(partial["status"], "partial")
        self.assertTrue(partial["healthy"])

    def _parser(self, feed_url: str):
        if "xbox" in feed_url:
            return SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        title="Xbox expands play across devices",
                        summary=(
                            "<p>Xbox confirmed an official Game Pass ecosystem "
                            "update for additional devices.</p> "
                            "The post Xbox expands play across devices "
                            "appeared first on Xbox Wire."
                        ),
                        link="https://news.xbox.com/es-latam/example/",
                        published_parsed=_parsed_date("2026-07-28"),
                    ),
                    SimpleNamespace(
                        title="Casino partnerships for esports betting",
                        summary=(
                            "The source announced a casino betting partnership "
                            "for an esports event."
                        ),
                        link="https://news.xbox.com/es-latam/blocked/",
                        published_parsed=_parsed_date("2026-07-28"),
                    ),
                ]
            )
        return SimpleNamespace(
            entries=[
                SimpleNamespace(
                    title="Google shares a new AI research update",
                    summary=(
                        "<p>Google published research that documents how "
                        "developers use the new tools.</p>"
                    ),
                    link="https://blog.google/technology/ai/example/",
                    published_parsed=_parsed_date("2026-07-27"),
                )
            ]
        )

    def test_collects_current_primary_sources_without_external_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates_path = root / "candidates.json"
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=candidates_path,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=self._parser,
            )
            candidates = json.loads(
                candidates_path.read_text(encoding="utf-8")
            )

        self.assertEqual(report["sources_scanned"], 2)
        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["rejection_counts"]["blocked_topic"], 1)
        self.assertFalse(report["publishing_attempted"])
        self.assertFalse(report["external_writes_attempted"])
        self.assertFalse(report["paid_generation_attempted"])
        self.assertFalse(report["media_download_attempted"])
        self.assertEqual(report["measured_cost_usd"], 0.0)
        self.assertEqual(
            {item["source_id"] for item in candidates},
            {"xbox_wire_es_latam", "google_blog"},
        )
        self.assertTrue(all(item["candidate_id"] for item in candidates))
        self.assertTrue(all(item["is_verified"] for item in candidates))
        self.assertTrue(
            all(item["strategic_classification"] for item in candidates)
        )
        self.assertTrue(
            all(
                item["strategic_classification"]["content_product_id"]
                for item in candidates
            )
        )
        self.assertTrue(
            all(
                item["strategic_classification"]["publishing_enabled"]
                is False
                for item in candidates
            )
        )
        self.assertGreater(candidates[0]["discovery_priority"], 0)
        xbox = next(
            item for item in candidates
            if item["source_id"] == "xbox_wire_es_latam"
        )
        self.assertEqual(
            xbox["summary"],
            (
                "Xbox confirmed an official Game Pass ecosystem update "
                "for additional devices."
            ),
        )

    def test_removes_spanish_feed_boilerplate(self):
        self.assert_feed_summary_is_cleaned(
            "<p>Game Pass llega a más dispositivos.</p> "
            "La entrada Game Pass llega a más dispositivos se publicó "
            "primero en Xbox Wire en Español.",
            "Game Pass llega a más dispositivos.",
        )

    def test_summary_only_boilerplate_is_blocked(self):
        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title="Xbox amplía Game Pass",
                summary=(
                    "The post Xbox amplía Game Pass appeared first on "
                    "Xbox Wire en Español ."
                ),
                link="https://news.xbox.com/es-latam/game-pass/",
                published_parsed=_parsed_date("2026-07-28"),
            )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(candidates, [])
        self.assertEqual(
            report["rejection_counts"]["missing_substantive_summary"],
            1,
        )

    def test_halo_game_pass_title_as_summary_is_blocked_and_next_survives(self):
        title = (
            "Próximamente en XBOX Game Pass: Halo: Campaign Evolved, "
            "Beast of Reincarnation y más"
        )

        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[
                SimpleNamespace(
                    title=title,
                    summary=title,
                    link="https://news.xbox.com/es-latam/halo-game-pass/",
                    published_parsed=_parsed_date("2026-07-28"),
                ),
                SimpleNamespace(
                    title="Xbox detalla una actualización de accesibilidad",
                    summary=(
                        "La actualización añade subtítulos configurables y "
                        "nuevos controles de contraste."
                    ),
                    link="https://news.xbox.com/es-latam/accessibility/",
                    published_parsed=_parsed_date("2026-07-28"),
                ),
            ])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["title"],
            "Xbox detalla una actualización de accesibilidad",
        )
        self.assertEqual(
            report["rejection_counts"]["summary_equivalent_to_title"],
            1,
        )

    def test_real_run_30399824136_visual_summaries_leave_empty_radar(self):
        fixture = json.loads(
            (
                ROOT
                / "fixtures"
                / "real_visual_summaries_run_30399824136.json"
            ).read_text(encoding="utf-8")
        )

        def parser(feed_url: str):
            if "google" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[
                SimpleNamespace(
                    title=item["title"],
                    summary=item["summary"],
                    link=f"https://blog.google/example-{index}/",
                    published_parsed=_parsed_date("2026-07-28"),
                )
                for index, item in enumerate(fixture)
            ])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(candidates, [])
        self.assertEqual(report["candidate_count"], 0)
        self.assertTrue(report["healthy"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(
            report["rejection_counts"]["summary_visual_metadata_only"],
            3,
        )
        self.assertEqual(
            report["rejection_counts"]["summary_placeholder"],
            1,
        )
        self.assertEqual(
            report["rejection_counts"][
                "summary_lacks_distinct_informative_proposition"
            ],
            1,
        )

    def test_real_run_30405453687_truncated_summary_is_rejected(self):
        def parser(feed_url: str):
            if "google" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title=(
                    "Experimenta el legado del Estadio Azteca "
                    "en Google Earth"
                ),
                summary=(
                    "El Estadio Azteca hizo historia durante el partido "
                    "inaugural. Al albergar este encuentro, se c…"
                ),
                link=(
                    "https://blog.google/intl/es-419/"
                    "actualizaciones-de-producto/informacion/"
                    "experimenta-el-legado-del-estadio-azteca/"
                ),
                published_parsed=_parsed_date("2026-07-28"),
            )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(candidates, [])
        self.assertEqual(
            report["rejection_counts"]["summary_truncated"],
            1,
        )

    def test_google_sports_technology_story_is_not_classified_as_ai(self):
        def parser(feed_url: str):
            if "google" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title=(
                    "Experimenta el legado del Estadio Azteca "
                    "en Google Earth"
                ),
                summary=(
                    "Google Earth incorpora un recorrido que documenta "
                    "momentos históricos del estadio y del fútbol mexicano."
                ),
                link=(
                    "https://blog.google/intl/es-419/"
                    "actualizaciones-de-producto/informacion/"
                    "experimenta-el-legado-del-estadio-azteca/"
                ),
                published_parsed=_parsed_date("2026-07-28"),
            )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["territory"],
            "sport_technology_entertainment",
        )

    def test_game_campaign_is_gaming_not_a_brand_activation(self):
        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title="Halo: Campaign Evolved",
                summary=(
                    "El remake incorpora una campaña cooperativa para cuatro "
                    "jugadores y añade modificadores de jugabilidad."
                ),
                link="https://news.xbox.com/es-latam/halo-campaign/",
                published_parsed=_parsed_date("2026-07-28"),
            )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["territory"], "gaming_esports")

    def test_truncated_feed_uses_bounded_article_evidence(self):
        fetch_calls = []

        def parser(feed_url: str):
            if "google" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title="Google Earth documenta el Estadio Azteca",
                summary="El recorrido digital se c…",
                link=(
                    "https://blog.google/intl/es-419/"
                    "actualizaciones-de-producto/informacion/azteca/"
                ),
                published_parsed=_parsed_date("2026-07-28"),
            )])

        def article_fetcher(source_url, allowed_domains):
            fetch_calls.append((source_url, allowed_domains))
            return (
                "<html><head><meta name='description' content='Google Earth "
                "incorpora un recorrido que documenta momentos históricos "
                "del Estadio Azteca y del fútbol mexicano.'></head></html>"
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
                article_fetcher=article_fetcher,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(fetch_calls), 1)
        self.assertEqual(report["article_fetch_attempts"], 1)
        self.assertEqual(report["article_fetch_successes"], 1)
        self.assertEqual(candidates[0]["summary_origin"], "article_page")
        self.assertEqual(
            candidates[0]["territory"],
            "sport_technology_entertainment",
        )
        self.assertFalse(report["publishing_attempted"])
        self.assertFalse(report["external_writes_attempted"])
        self.assertFalse(report["media_download_attempted"])

    def test_article_enrichment_respects_per_source_limit(self):
        def parser(feed_url: str):
            if "google" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[
                SimpleNamespace(
                    title=f"Historia incompleta {index}",
                    summary="Resumen incompleto…",
                    link=f"https://blog.google/example-{index}/",
                    published_parsed=_parsed_date("2026-07-28"),
                )
                for index in range(4)
            ])

        fetch_calls = []

        def article_fetcher(source_url, _allowed_domains):
            fetch_calls.append(source_url)
            return "<html><meta name='description' content='Sin datos…'></html>"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=root / "candidates.json",
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
                article_fetcher=article_fetcher,
                max_article_fetches_per_source=2,
            )

        self.assertEqual(len(fetch_calls), 2)
        self.assertEqual(report["article_fetch_attempts"], 2)
        self.assertEqual(report["article_fetch_successes"], 0)

    def test_routine_xbox_roundup_is_rejected_before_article_fetch(self):
        fetch_calls = []

        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title=(
                    "La Próxima Semana en XBOX: nuevos juegos "
                    "del 27 al 31 de julio"
                ),
                summary="Nuevos lanzamientos…",
                link="https://news.xbox.com/es-latam/weekly-roundup/",
                published_parsed=_parsed_date("2026-07-28"),
            )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=root / "candidates.json",
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
                article_fetcher=lambda *_args: fetch_calls.append(True),
            )

        self.assertEqual(fetch_calls, [])
        self.assertEqual(
            report["rejection_counts"]["routine_release_roundup"],
            1,
        )

    def test_broad_google_source_requires_a_real_territory_signal(self):
        def parser(feed_url: str):
            if "google" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title="Nueva opción para acceder a tu cuenta",
                summary=(
                    "El video selfie ofrece otra forma de iniciar sesión "
                    "cuando no tienes tu dispositivo habitual."
                ),
                link="https://blog.google/intl/es-419/account-access/",
                published_parsed=_parsed_date("2026-07-28"),
            )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=root / "candidates.json",
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )

        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(
            report["rejection_counts"]["outside_editorial_territory"],
            1,
        )

    def test_source_report_discloses_article_read_mode(self):
        def parser(feed_url: str):
            if "google" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title="Nuevo análisis sobre inteligencia artificial",
                summary="Análisis incompleto…",
                link="https://blog.google/intl/es-419/ai-analysis/",
                published_parsed=_parsed_date("2026-07-28"),
            )])

        def article_fetcher(*_args):
            return (
                "<meta name='description' content='El informe analiza cómo "
                "las empresas usan inteligencia artificial en 20 países.'>"
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=root / "candidates.json",
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
                article_fetcher=article_fetcher,
            )

        google = next(
            source for source in report["sources"]
            if source["source_id"] == "google_blog"
        )
        self.assertEqual(
            google["network_mode"],
            "rss_and_approved_article_read_only",
        )

    def test_real_halo_intro_cannot_satisfy_feature_headline(self):
        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title=(
                    "Halo: Campaign Evolved – Las nuevas funciones "
                    "del remake de un clásico"
                ),
                summary="Imagen promocional de Halo.",
                link=(
                    "https://news.xbox.com/es-latam/2026/07/23/"
                    "halo-campaign-evolved/"
                ),
                published_parsed=_parsed_date("2026-07-28"),
            )])

        def article_fetcher(*_args):
            return (
                "<meta name='description' content='Desde los pasillos del "
                "Covenant hasta las llanuras del anillo, estos escenarios "
                "marcaron el inicio del viaje de muchas personas y esta "
                "edición ofrece mucho más que nostalgia.'>"
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=root / "candidates.json",
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
                article_fetcher=article_fetcher,
            )

        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(
            report["rejection_counts"][
                "summary_does_not_fulfill_title_promise"
            ],
            1,
        )

    def assert_feed_summary_is_cleaned(self, summary, expected):
        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[SimpleNamespace(
                title="Xbox amplía Game Pass",
                summary=summary,
                link="https://news.xbox.com/es-latam/game-pass/",
                published_parsed=_parsed_date("2026-07-28"),
            )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(candidates[0]["summary"], expected)

    def test_same_story_with_alias_urls_is_rejected_as_duplicate(self):
        def parser(feed_url: str):
            if "xbox" in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(entries=[
                SimpleNamespace(
                    title="Experimenta el legado del Estadio Azteca",
                    summary=(
                        "Google Earth incorpora un recorrido que documenta "
                        "momentos históricos del estadio."
                    ),
                    link=(
                        "https://blog.google/intl/es-419/"
                        "actualizaciones-de-producto/informacion/"
                        "experimenta-el-legado-del-estadio-azteca/"
                    ),
                    published_parsed=_parsed_date("2026-07-28"),
                ),
                SimpleNamespace(
                    title="EXPERIMENTA EL LEGADO DEL ESTADIO AZTECA",
                    summary=(
                        "Google publicó la misma experiencia desde un alias "
                        "del feed oficial."
                    ),
                    link=(
                        "https://blog.google/intl/es-419/feed/"
                        "experimenta-el-legado-del-estadio-azteca/"
                    ),
                    published_parsed=_parsed_date("2026-07-28"),
                ),
            ])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["rejection_counts"]["duplicate_story"], 1)
        self.assertEqual(len(candidates), 1)
        google = next(
            source for source in report["sources"]
            if source["source_id"] == "google_blog"
        )
        self.assertEqual(google["accepted"], 1)
        self.assertEqual(google["rejected"], 1)

    def test_substantive_platform_story_outranks_a_promotion(self):
        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        title="Win a special Halo-inspired collectible",
                        summary=(
                            "Xbox announced a limited promotion for players "
                            "in selected regions."
                        ),
                        link="https://news.xbox.com/es-latam/promotion/",
                        published_parsed=_parsed_date("2026-07-28"),
                    ),
                    SimpleNamespace(
                        title="Xbox expands backward compatibility on PC",
                        summary=(
                            "Xbox confirmed that additional games now work "
                            "through backward compatibility on PC."
                        ),
                        link="https://news.xbox.com/es-latam/platform-update/",
                        published_parsed=_parsed_date("2026-07-27"),
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            candidates[0]["title"],
            "Xbox expands backward compatibility on PC",
        )
        self.assertGreater(
            candidates[0]["discovery_priority"],
            candidates[1]["discovery_priority"],
        )

    def test_candidate_ids_and_order_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            for output in (first, second):
                collect_live_candidates(
                    registry_path=SOURCES_PATH,
                    output_path=output,
                    report_path=root / f"{output.stem}-report.json",
                    today=date(2026, 7, 28),
                    parser=self._parser,
                )

            self.assertEqual(
                json.loads(first.read_text(encoding="utf-8")),
                json.loads(second.read_text(encoding="utf-8")),
            )

    def test_rejects_stale_and_wrong_domain_entries(self):
        def parser(feed_url: str):
            return SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        title="Old story",
                        summary=(
                            "The source confirmed this product update during "
                            "the previous month."
                        ),
                        link=(
                            "https://news.xbox.com/es-latam/old/"
                            if "xbox" in feed_url
                            else "https://example.com/wrong/"
                        ),
                        published_parsed=_parsed_date("2026-06-01"),
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect_live_candidates(
                registry_path=SOURCES_PATH,
                output_path=root / "candidates.json",
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                parser=parser,
            )

        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["rejection_counts"]["stale_story"], 1)
        self.assertEqual(
            report["rejection_counts"]["source_domain_mismatch"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
