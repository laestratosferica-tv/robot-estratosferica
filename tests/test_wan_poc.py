import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from wan.client import ComfyClient, load_json, prepare_workflow


ROOT = Path(__file__).resolve().parents[1]


class FakeComfyHandler(BaseHTTPRequestHandler):
    prompt_id = "mock-prompt-1"

    def log_message(self, *_args):
        return

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/system_stats":
            return self._json({"system": {"os": "mock"}})
        if self.path == f"/history/{self.prompt_id}":
            return self._json({self.prompt_id: {"status": {"status_str": "success"}, "outputs": {"47": {"videos": [{"filename": "wan.webm", "subfolder": "", "type": "output"}]}}}})
        if self.path.startswith("/view?"):
            payload = b"fake-webm"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return self.wfile.write(payload)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(size)
        if self.path == "/upload/image":
            return self._json({"name": "reference.png"})
        if self.path == "/prompt":
            return self._json({"prompt_id": self.prompt_id})
        return self._json({"error": "not found"}, 404)


class WanPocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfyHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_configuration_is_safe_by_default(self):
        config = load_json(ROOT / "wan/config/wan_poc.json")
        self.assertFalse(config["enabled"])
        self.assertFalse(config["allow_paid_remote"])
        self.assertEqual(config["server_url"], "")

    def test_workflow_injection(self):
        template = load_json(ROOT / "wan/workflows/wan22_ti2v_5b_api.json")
        result = prepare_workflow(template, "avatar.png", "hola", "mal", 720, 1280, 61, 25, 22, 4.5, 9)
        self.assertEqual(result["57"]["inputs"]["image"], "avatar.png")
        self.assertEqual(result["6"]["inputs"]["text"], "hola")
        self.assertEqual(result["55"]["inputs"]["length"], 61)
        self.assertEqual(result["3"]["inputs"]["seed"], 9)
        self.assertEqual(result["47"]["inputs"]["fps"], 25)

    def test_end_to_end_against_mock_comfyui(self):
        client = ComfyClient(self.url)
        self.assertEqual(client.health()["system"]["os"], "mock")
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "reference.png"
            image.write_bytes(b"png")
            self.assertEqual(client.upload_image(image), "reference.png")
            prompt_id = client.queue_prompt({"1": {"class_type": "Mock", "inputs": {}}})
            history, polls = client.wait_for_output(prompt_id, poll_seconds=0.01, max_wait_seconds=2)
            self.assertGreaterEqual(polls, 1)
            output = client.find_output(history)
            destination = client.download_output(output, Path(temp_dir) / "result.webm")
            self.assertEqual(destination.read_bytes(), b"fake-webm")


if __name__ == "__main__":
    unittest.main()
