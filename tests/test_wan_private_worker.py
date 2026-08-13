import base64
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HANDLER_PATH = ROOT / "wan/private_worker/handler.py"
spec = importlib.util.spec_from_file_location("wan_official_handler", HANDLER_PATH)
handler = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(handler)


class WanPrivateWorkerTests(unittest.TestCase):
    def valid_input(self):
        return {
            "prompt": "Movimiento natural del personaje",
            "image_base64": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(),
            "frame_num": 17,
            "sample_steps": 10,
            "seed": 72993276,
        }

    def test_dockerfile_uses_only_official_pinned_sources(self):
        dockerfile = (ROOT / "wan/private_worker/Dockerfile").read_text()
        self.assertIn("github.com/Wan-Video/Wan2.2.git", dockerfile)
        self.assertIn("42bf4cfaa384bc21833865abc2f9e6c0e67233dc", dockerfile)
        self.assertIn("Wan-AI/Wan2.2-TI2V-5B", dockerfile)
        self.assertNotIn("averystorm", dockerfile.casefold())
        self.assertNotIn("wlsdml", dockerfile.casefold())
        self.assertNotIn(":latest", dockerfile)

    def test_input_is_private_base64_only(self):
        with self.assertRaisesRegex(ValueError, "image_base64 is required"):
            handler.validate_input({"prompt": "x", "image_url": "https://example.com/a.png"})

    def test_input_rejects_invalid_media_and_unsafe_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "PNG or JPEG"):
                handler._decode_image(base64.b64encode(b"text").decode(), Path(temp_dir) / "x")
        invalid = self.valid_input() | {"size": "1920*1080"}
        with self.assertRaisesRegex(ValueError, "size must be"):
            handler.validate_input(invalid)

    def test_frame_count_is_bounded_and_matches_4n_plus_1(self):
        for value in (16, 122):
            invalid = self.valid_input() | {"frame_num": value}
            with self.assertRaises(ValueError):
                handler.validate_input(invalid)

    def test_command_uses_official_ti2v_memory_flags(self):
        validated = handler.validate_input(self.valid_input())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            model = root / "model"
            source.mkdir()
            model.mkdir()
            (source / "generate.py").write_text("# official")
            with patch.dict(os.environ, {"WAN_SOURCE_DIR": str(source), "WAN_MODEL_DIR": str(model)}):
                command = handler.build_command(validated, root / "input.png", root / "output.mp4")
        self.assertIn("ti2v-5B", command)
        self.assertIn("--offload_model", command)
        self.assertIn("--convert_model_dtype", command)
        self.assertIn("--t5_cpu", command)
        self.assertNotIn("--use_prompt_extend", command)

    def test_self_test_does_not_load_model_or_gpu(self):
        self.assertEqual(handler.self_test(), 0)


if __name__ == "__main__":
    unittest.main()
