"""Cliente seguro para voces privadas de ElevenLabs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_NAME = "Joseverso — Privada"
DEFAULT_MODEL = "eleven_multilingual_v2"


class ElevenLabsConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceSettings:
    speed: float = 1.0
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True

    def as_payload(self) -> dict[str, Any]:
        return {
            "speed": self.speed,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }


class ElevenLabsTTS:
    def __init__(self, api_key: Optional[str] = None, *, session: Any = None, timeout_seconds: int = 60) -> None:
        self._api_key = (api_key or os.getenv("ELEVENLABS_API_KEY", "")).strip()
        if session is None:
            import requests

            session = requests
        self._session = session
        self._timeout = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _headers(self, *, json_response: bool = True) -> dict[str, str]:
        if not self.configured:
            raise ElevenLabsConfigurationError(
                "Falta ELEVENLABS_API_KEY; debe vivir en secretos, no en archivos."
            )
        headers = {"xi-api-key": self._api_key}
        if json_response:
            headers["Accept"] = "application/json"
        return headers

    def resolve_private_voice_id(self, voice_name: str = DEFAULT_VOICE_NAME) -> str:
        response = self._session.get(
            f"{API_BASE}/voices", headers=self._headers(), timeout=self._timeout
        )
        response.raise_for_status()
        matches = [v for v in response.json().get("voices", []) if v.get("name") == voice_name]
        if len(matches) != 1:
            raise ElevenLabsConfigurationError(
                f"Se esperaba una voz privada llamada {voice_name!r}; encontradas: {len(matches)}."
            )
        voice_id = str(matches[0].get("voice_id", "")).strip()
        if not voice_id:
            raise ElevenLabsConfigurationError("La voz privada no tiene voice_id válido.")
        return voice_id

    def synthesize_approved_script(
        self,
        text: str,
        *,
        approval_id: str,
        voice_name: str = DEFAULT_VOICE_NAME,
        model_id: str = DEFAULT_MODEL,
        settings: VoiceSettings = VoiceSettings(),
    ) -> bytes:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("El guion está vacío.")
        if not (approval_id or "").strip():
            raise PermissionError("Joseverso solo puede leer guiones aprobados.")

        voice_id = self.resolve_private_voice_id(voice_name)
        response = self._session.post(
            f"{API_BASE}/text-to-speech/{voice_id}",
            headers={
                **self._headers(json_response=False),
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            json={
                "text": clean_text,
                "model_id": model_id,
                "voice_settings": settings.as_payload(),
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.content


def safe_voice_status(env: Mapping[str, str] = os.environ) -> dict[str, Any]:
    return {
        "provider": "ElevenLabs",
        "voice_name": DEFAULT_VOICE_NAME,
        "configured": bool((env.get("ELEVENLABS_API_KEY") or "").strip()),
        "publishing_enabled": False,
        "requires_approved_script": True,
    }
