import unittest

from threads_reply_collector import ThreadsReadOnlyClient, collect_and_classify


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payloads.pop(0))


class ThreadsReplyCollectorTests(unittest.TestCase):
    def test_client_uses_get_only_and_keeps_token_in_query(self):
        session = FakeSession([{"data": [{"id": "p1"}]}])
        client = ThreadsReadOnlyClient("secret", session=session)
        posts = client.get_own_threads(limit=1)
        self.assertEqual(posts, [{"id": "p1"}])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]["params"]["access_token"], "secret")

    def test_collects_and_prioritizes_a_commercial_reply(self):
        session = FakeSession(
            [
                {"data": [{"id": "p1", "permalink": "https://threads.net/p1"}]},
                {
                    "data": [
                        {
                            "id": "r1",
                            "text": "Nuestra marca quiere una cotización de pauta",
                            "username": "marca_demo",
                            "timestamp": "2026-07-26T12:00:00+0000",
                            "permalink": "https://threads.net/r1",
                        }
                    ]
                },
            ]
        )
        report = collect_and_classify(
            ThreadsReadOnlyClient("secret", session=session),
            thread_limit=1,
        )
        self.assertEqual(report["mode"], "analysis_only")
        self.assertFalse(report["outbound_actions_enabled"])
        self.assertEqual(report["counts"]["commercial_lead"], 1)
        self.assertEqual(report["queue"][0]["username"], "marca_demo")
        self.assertFalse(report["queue"][0]["can_auto_send"])


if __name__ == "__main__":
    unittest.main()
