import json
import tempfile
import time
import unittest
from unittest.mock import patch
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

from live_source_radar import collect_live_candidates, main


ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "config" / "sources_v1.json"


def _parsed_date(value: str) -> time.struct_time:
    return time.strptime(value, "%Y-%m-%d")


class LiveSourceRadarTests(unittest.TestCase):
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
                            "<p>An official Game Pass ecosystem update.</p> "
                            "The post Xbox expands play across devices "
                            "appeared first on Xbox Wire."
                        ),
                        link="https://news.xbox.com/es-latam/example/",
                        published_parsed=_parsed_date("2026-07-28"),
                    ),
                    SimpleNamespace(
                        title="Casino partnerships for esports betting",
                        summary="Blocked topic",
                        link="https://news.xbox.com/es-latam/blocked/",
                        published_parsed=_parsed_date("2026-07-28"),
                    ),
                ]
            )
        return SimpleNamespace(
            entries=[
                SimpleNamespace(
                    title="Google shares a new AI research update",
                    summary="<p>New research and developer tools.</p>",
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
            "An official Game Pass ecosystem update.",
        )

    def test_removes_spanish_feed_boilerplate(self):
        self.assert_feed_summary_is_cleaned(
            "<p>Game Pass llega a más dispositivos.</p> "
            "La entrada Game Pass llega a más dispositivos se publicó "
            "primero en Xbox Wire en Español.",
            "Game Pass llega a más dispositivos.",
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

    def test_substantive_platform_story_outranks_a_promotion(self):
        def parser(feed_url: str):
            if "xbox" not in feed_url:
                return SimpleNamespace(entries=[])
            return SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        title="Win a special Halo-inspired collectible",
                        summary="A limited promotion.",
                        link="https://news.xbox.com/es-latam/promotion/",
                        published_parsed=_parsed_date("2026-07-28"),
                    ),
                    SimpleNamespace(
                        title="Xbox expands backward compatibility on PC",
                        summary="A platform and ecosystem update.",
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
                        summary="No longer current",
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
