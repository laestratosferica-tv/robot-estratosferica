import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from epic_play_radar import collect_epic_play_candidates


class EpicPlayRadarTests(unittest.TestCase):
    def test_scores_candidates_and_keeps_them_link_only(self):
        twitch = [
            {
                "id": "clip-1",
                "title": "ACE imposible en la final de VALORANT",
                "creator_name": "Creadora LATAM",
                "creator_url": "https://www.twitch.tv/creadora",
                "game_name": "VALORANT",
                "url": "https://clips.twitch.tv/clip-1",
                "created_at": "2026-07-28T10:00:00Z",
                "view_count": 60000,
            }
        ]
        youtube = [
            {
                "id": "video-1",
                "title": "Highlights de una partida competitiva",
                "channel": "Canal Gamer",
                "game": "Rocket League",
                "url": "https://www.youtube.com/watch?v=video-1",
                "published_at": "2026-07-27T18:00:00Z",
                "view_count": 12000,
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_epic_play_candidates(
                providers={
                    "twitch": lambda: twitch,
                    "youtube": lambda: youtube,
                },
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(candidates[0]["external_id"], "clip-1")
        self.assertEqual(candidates[0]["rights"]["state"], "link_only_unverified")
        self.assertFalse(candidates[0]["rights"]["reuse_allowed"])
        self.assertFalse(candidates[0]["rights"]["download_allowed"])
        self.assertFalse(candidates[0]["workflow"]["automatic_publish_allowed"])
        self.assertTrue(candidates[0]["workflow"]["human_approval_required"])
        strategy = candidates[0]["strategic_classification"]
        self.assertEqual(
            strategy["content_product_id"],
            "jugada_estratosferica",
        )
        self.assertEqual(
            strategy["rights_state"],
            "official_embed_or_link",
        )
        self.assertFalse(strategy["publishing_enabled"])
        self.assertFalse(strategy["broadcasting_enabled"])
        self.assertEqual(candidates[0]["creator_name"], "Creadora LATAM")
        self.assertEqual(
            candidates[0]["source_url"],
            "https://clips.twitch.tv/clip-1",
        )
        self.assertFalse(report["publishing_attempted"])
        self.assertFalse(report["media_download_attempted"])
        self.assertFalse(report["rights_assumed"])
        self.assertEqual(report["measured_cost_usd"], 0.0)

    def test_rejects_old_invalid_and_duplicate_candidates(self):
        items = [
            {
                "id": "same",
                "title": "Clutch en torneo",
                "creator_name": "Jugador",
                "url": "https://clips.twitch.tv/same",
                "created_at": "2026-07-28T10:00:00Z",
                "view_count": 5000,
            },
            {
                "id": "same",
                "title": "El mismo clip",
                "creator_name": "Jugador",
                "url": "https://clips.twitch.tv/same-copy",
                "created_at": "2026-07-28T10:00:00Z",
                "view_count": 5000,
            },
            {
                "id": "old",
                "title": "Una jugada vieja",
                "creator_name": "Jugador",
                "url": "https://clips.twitch.tv/old",
                "created_at": "2026-06-01T10:00:00Z",
                "view_count": 5000,
            },
            {
                "id": "no-creator",
                "title": "Sin crédito",
                "creator_name": "",
                "url": "https://clips.twitch.tv/no-creator",
                "created_at": "2026-07-28T10:00:00Z",
                "view_count": 5000,
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_epic_play_candidates(
                providers={"twitch": lambda: items},
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(report["rejection_counts"]["duplicate_candidate"], 1)
        self.assertEqual(report["rejection_counts"]["candidate_too_old"], 1)
        self.assertEqual(
            report["rejection_counts"]["candidate_missing_creator"],
            1,
        )

    def test_caps_results_after_scoring(self):
        def candidate(number, views):
            return {
                "id": f"video-{number}",
                "title": f"Highlights {number}",
                "channel": "Canal",
                "url": f"https://youtube.com/watch?v={number}",
                "published_at": "2026-07-28T10:00:00Z",
                "view_count": views,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.json"
            report = collect_epic_play_candidates(
                providers={
                    "youtube": lambda: [
                        candidate(1, 500),
                        candidate(2, 50000),
                        candidate(3, 10000),
                    ]
                },
                output_path=output,
                report_path=root / "report.json",
                today=date(2026, 7, 28),
                max_candidates=2,
            )
            candidates = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(
            [item["external_id"] for item in candidates],
            ["video-2", "video-3"],
        )
