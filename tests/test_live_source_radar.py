import json
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from live_source_radar import collect_live_candidates


ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "config" / "sources_v1.json"


def _parsed_date(value: str) -> time.struct_time:
    return time.strptime(value, "%Y-%m-%d")


class LiveSourceRadarTests(unittest.TestCase):
    def _parser(self, feed_url: str):
        if "xbox" in feed_url:
            return SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        title="Xbox expands play across devices",
                        summary=(
                            "<p>An official Game Pass ecosystem update.</p>"
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
