import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER_PATH = ROOT / "wan/private_worker/harden_worker.py"

spec = importlib.util.spec_from_file_location("harden_worker", PATCHER_PATH)
harden_worker = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(harden_worker)


class WanPrivateWorkerTests(unittest.TestCase):
    def sample_handler(self):
        return "\n".join(
            [
                "def handler(job):",
                "    job_input = job.get(\"input\", {})",
                harden_worker.UNSAFE_LOG,
                harden_worker.UNSAFE_IMAGE_INPUT,
                harden_worker.UNSAFE_END_INPUT,
                "def save_base64_to_file(base64_data):",
                harden_worker.UNSAFE_DECODE,
            ]
        )

    def test_hardening_removes_sensitive_logging_and_remote_inputs(self):
        result = harden_worker.harden_text(self.sample_handler())
        self.assertNotIn("{job_input}", result)
        self.assertNotIn('"image_url" in job_input', result)
        self.assertNotIn('"image_path" in job_input', result)
        self.assertIn("image_base64 is required", result)
        self.assertIn("validate=True", result)
        self.assertIn("Decoded image exceeds 9 MiB", result)

    def test_hardening_fails_closed_when_upstream_changes(self):
        with self.assertRaises(RuntimeError):
            harden_worker.harden_text("def handler(job): pass")

    def test_dockerfile_uses_pinned_public_sources(self):
        dockerfile = (ROOT / "wan/private_worker/Dockerfile").read_text()
        self.assertNotIn("registry.runpod.net", dockerfile)
        self.assertIn("FROM wlsdml1114/engui_genai-base_blackwell:1.1", dockerfile)
        self.assertIn("UPSTREAM_WORKER_COMMIT=4b6d5ec27dae6409bd2011a96d8e819e67d4ebaa", dockerfile)
        self.assertIn("COMFYUI_COMMIT=ddbaa8752874c275290d054ee4fddd6e004f5fdf", dockerfile)
        self.assertIn("py_compile", dockerfile)
        self.assertNotIn(":latest", dockerfile)

    def test_script_writes_only_target_handler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = Path(temp_dir) / "handler.py"
            handler.write_text(self.sample_handler())
            handler.write_text(harden_worker.harden_text(handler.read_text()))
            self.assertIn("Received job input keys", handler.read_text())


if __name__ == "__main__":
    unittest.main()
