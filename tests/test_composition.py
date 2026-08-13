import unittest
from pathlib import Path

from media_factory.composition import build_composition_plan, load_composition_config


ROOT = Path(__file__).resolve().parents[1]


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_composition_config(ROOT / "config/composition_v1.json")

    def test_complete_plan_is_ready_but_render_stays_disabled(self):
        plan = build_composition_plan({
            "content_id": "demo-1",
            "presenter_video": "heygen.mp4",
            "context_videos": ["gameplay.mp4"],
            "wan_scenes": [],
            "graphics_manifest": "hyperframes.json",
            "captions_file": "captions.srt",
            "source_credit": "Fuente: estudio oficial",
            "output_file": "final.mp4",
        }, self.config)
        self.assertEqual(plan.state, "ready_for_render")
        self.assertFalse(plan.render_enabled)

    def test_missing_presenter_captions_and_credit_are_blocked(self):
        plan = build_composition_plan({"output_file": "final.mp4"}, self.config)
        self.assertEqual(plan.state, "blocked")
        self.assertIn("missing_heygen_presenter", plan.blockers)
        self.assertIn("missing_captions", plan.blockers)
        self.assertIn("missing_source_credit", plan.blockers)

    def test_output_must_be_mp4(self):
        plan = build_composition_plan({
            "presenter_video": "heygen.mp4", "captions_file": "captions.srt",
            "source_credit": "Fuente", "output_file": "final.mov",
        }, self.config)
        self.assertIn("output_must_be_mp4", plan.blockers)


if __name__ == "__main__":
    unittest.main()
