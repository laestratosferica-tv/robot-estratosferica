import unittest
from datetime import datetime, timezone

from epic_play_providers import (
    ProviderConfigurationError,
    twitch_clip_provider,
    youtube_video_provider,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return FakeResponse(next(self.responses))

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return FakeResponse(next(self.responses))


class EpicPlayProviderTests(unittest.TestCase):
    def test_twitch_reads_metadata_without_download_endpoint(self):
        session = FakeSession(
            [
                {"access_token": "temporary"},
                {"data": [{"id": "516575", "name": "VALORANT"}]},
                {
                    "data": [
                        {
                            "id": "clip-1",
                            "title": "ACE imposible",
                            "broadcaster_name": "CreadoraLatam",
                            "url": "https://clips.twitch.tv/clip-1",
                            "created_at": "2026-07-28T10:00:00Z",
                            "view_count": 12345,
                        }
                    ]
                },
            ]
        )
        items = twitch_clip_provider(
            game_names=["VALORANT"],
            env={
                "TWITCH_CLIENT_ID": "client",
                "TWITCH_CLIENT_SECRET": "secret",
            },
            session=session,
            now=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(items[0]["creator_name"], "CreadoraLatam")
        self.assertEqual(items[0]["game_name"], "VALORANT")
        urls = [call[1] for call in session.calls]
        self.assertIn("https://api.twitch.tv/helix/clips", urls)
        self.assertFalse(any("download" in url for url in urls))

    def test_youtube_caps_searches_and_fetches_statistics(self):
        session = FakeSession(
            [
                {"access_token": "temporary"},
                {
                    "items": [
                        {
                            "id": {"videoId": "video-1"},
                            "snippet": {"title": "resultado parcial"},
                        }
                    ]
                },
                {
                    "items": [
                        {
                            "id": "video-1",
                            "snippet": {
                                "title": "Clutch en la final",
                                "description": "Jugada competitiva",
                                "channelTitle": "Canal LATAM",
                                "channelId": "channel-1",
                                "publishedAt": "2026-07-28T10:00:00Z",
                            },
                            "statistics": {"viewCount": "42000"},
                        }
                    ]
                },
            ]
        )
        items = youtube_video_provider(
            queries=["query 1", "query 2", "query 3"],
            env={
                "YOUTUBE_CLIENT_ID": "client",
                "YOUTUBE_CLIENT_SECRET": "secret",
                "YOUTUBE_REFRESH_TOKEN": "refresh",
            },
            session=session,
            now=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
            max_queries=1,
        )

        search_calls = [
            call for call in session.calls if call[1].endswith("/search")
        ]
        self.assertEqual(len(search_calls), 1)
        self.assertEqual(items[0]["view_count"], "42000")
        self.assertEqual(
            items[0]["url"],
            "https://www.youtube.com/watch?v=video-1",
        )

    def test_missing_secrets_fail_before_network_access(self):
        session = FakeSession([])
        with self.assertRaisesRegex(
            ProviderConfigurationError,
            "missing_twitch_client_secret",
        ):
            twitch_clip_provider(
                game_names=["VALORANT"],
                env={"TWITCH_CLIENT_ID": "client"},
                session=session,
            )
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
