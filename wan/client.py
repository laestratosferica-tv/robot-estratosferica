from __future__ import annotations

import copy
import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass
class GenerationStats:
    prompt_id: str
    elapsed_seconds: float
    poll_count: int
    output_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComfyClient:
    def __init__(self, base_url: str, token: str = "", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _request(self, method: str, path: str, body: bytes | None = None,
                 headers: dict[str, str] | None = None) -> bytes:
        request = Request(
            f"{self.base_url}{path}", data=body, method=method,
            headers={**self.headers, **(headers or {})},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code} en {path}: {details}") from error

    def _json_request(self, method: str, path: str,
                      payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        raw = self._request(method, path, body, {"Content-Type": "application/json"})
        return json.loads(raw.decode())

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", "/system_stats")

    def upload_image(self, image_path: Path) -> str:
        boundary = f"----wanpoc{uuid.uuid4().hex}"
        content = image_path.read_bytes()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{image_path.name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        payload = json.loads(self._request(
            "POST", "/upload/image", body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        ).decode())
        return payload.get("name", image_path.name)

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        return self._json_request(
            "POST", "/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())}
        )["prompt_id"]

    def wait_for_output(
        self, prompt_id: str, poll_seconds: float = 2.0, max_wait_seconds: int = 3600
    ) -> tuple[dict[str, Any], int]:
        started = time.monotonic()
        polls = 0
        while time.monotonic() - started < max_wait_seconds:
            polls += 1
            history = self._json_request("GET", f"/history/{prompt_id}")
            if prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI reportó error: {status}")
                if item.get("outputs"):
                    return item, polls
            time.sleep(poll_seconds)
        raise TimeoutError(f"ComfyUI no terminó en {max_wait_seconds} segundos")

    @staticmethod
    def find_output(history_item: dict[str, Any]) -> dict[str, str]:
        for node in history_item.get("outputs", {}).values():
            for key in ("videos", "gifs", "images"):
                files = node.get(key, [])
                if files:
                    return files[0]
        raise RuntimeError("La ejecución terminó sin archivo descargable")

    def download_output(self, output: dict[str, str], destination: Path) -> Path:
        query = urlencode({
            "filename": output["filename"],
            "subfolder": output.get("subfolder", ""),
            "type": output.get("type", "output"),
        })
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._request("GET", f"/view?{query}"))
        return destination


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_workflow(
    template: dict[str, Any], image_name: str, prompt: str, negative_prompt: str,
    width: int, height: int, frames: int, fps: int, steps: int, cfg: float, seed: int,
) -> dict[str, Any]:
    workflow = copy.deepcopy(template)
    workflow["57"]["inputs"]["image"] = image_name
    workflow["6"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["55"]["inputs"].update({"width": width, "height": height, "length": frames})
    workflow["3"]["inputs"].update({"steps": steps, "cfg": cfg, "seed": seed})
    workflow["47"]["inputs"]["fps"] = fps
    return workflow


def ensure_mp4(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(destination)],
        check=True, capture_output=True, text=True,
    )
    return destination
