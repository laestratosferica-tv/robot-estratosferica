import unittest

from media_factory.elevenlabs_tts import ElevenLabsConfigurationError, ElevenLabsTTS, VoiceSettings, safe_voice_status


class FakeResponse:
    def __init__(self, *, payload=None, content=b"audio"):
        self._payload = payload or {}
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.posts = []

    def get(self, url, **kwargs):
        return FakeResponse(payload={"voices": [{"name": "Joseverso — Privada", "voice_id": "private-1"}]})

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return FakeResponse(content=b"approved-audio")


class ElevenLabsTTSTests(unittest.TestCase):
    def test_status_never_exposes_secret(self):
        status = safe_voice_status({"ELEVENLABS_API_KEY": "secret-value"})
        self.assertTrue(status["configured"])
        self.assertNotIn("secret-value", repr(status))
        self.assertFalse(status["publishing_enabled"])

    def test_missing_secret_fails_closed(self):
        with self.assertRaises(ElevenLabsConfigurationError):
            ElevenLabsTTS(api_key="", session=FakeSession()).resolve_private_voice_id()

    def test_script_requires_approval(self):
        with self.assertRaises(PermissionError):
            ElevenLabsTTS(api_key="x", session=FakeSession()).synthesize_approved_script("Hola", approval_id="")

    def test_approved_script_uses_locked_voice_settings(self):
        session = FakeSession()
        audio = ElevenLabsTTS(api_key="x", session=session).synthesize_approved_script(
            "Hola desde Joseverso", approval_id="approved-001"
        )
        self.assertEqual(audio, b"approved-audio")
        _, request = session.posts[0]
        self.assertEqual(request["json"]["model_id"], "eleven_multilingual_v2")
        self.assertEqual(request["json"]["language_code"], "es")
        self.assertEqual(request["json"]["voice_settings"], VoiceSettings().as_payload())
        self.assertNotIn("approval_id", request["json"])


if __name__ == "__main__":
    unittest.main()
