from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


@dataclass
class RunpodGenerationStats:
    job_id: str
    elapsed_seconds: float
    poll_count: int
    output_file: str
    endpoint_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunpodWanClient:
    """Cliente pequeño para un worker privado de Wan en RunPod Serverless."""

    def __init__(self, endpoint_id: str, api_key: str, timeout: int = 60):
        if not endpoint_id:
            raise ValueError("Falta RUNPOD_ENDPOINT_ID")
        if not api_key:
            raise ValueError("Falta RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id
        self.timeout = timeout
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _json_request(self, method: str, path: str,
                      payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(f"{self.base_url}{path}", data=body, method=method, headers=self.headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod HTTP {error.code} en {path}: {detail}") from error

    @staticmethod
    def encode_image(image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("ascii")

    def submit(self, payload: dict[str, Any]) -> str:
        response = self._json_request("POST", "/run", {"input": payload})
        job_id = response.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod no devolvió job id: {response}")
        return job_id

    def wait(self, job_id: str, poll_seconds: float = 2.0,
             max_wait_seconds: int = 3600) -> tuple[dict[str, Any], int]:
        started = time.monotonic()
        polls = 0
        while time.monotonic() - started < max_wait_seconds:
            polls += 1
            status = self._json_request("GET", f"/status/{job_id}")
            state = status.get("status")
            if state in TERMINAL_STATES:
                if state != "COMPLETED":
                    raise RuntimeError(f"RunPod terminó en {state}: {status.get('error', '')}")
                return status, polls
            time.sleep(poll_seconds)
        raise TimeoutError(f"RunPod no terminó en {max_wait_seconds} segundos")

    @staticmethod
    def save_video(status: dict[str, Any], destination: Path) -> Path:
        output = status.get("output", {})
        video = output.get("video") if isinstance(output, dict) else None
        if not video:
            raise RuntimeError("El worker terminó sin output.video")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(video, validate=True))
        return destination
