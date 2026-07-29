import unittest

from media_factory.creative_variation import (
    load_creative_variation,
    select_creative_profile,
    validate_creative_profile,
    validate_creative_variation,
)
from media_factory.models import Candidate


class CreativeVariationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_creative_variation()
        self.candidate = Candidate(
            candidate_id="story-1",
            title="Cambio competitivo confirmado",
            source_url="https://example.com/story",
            territory="gaming_esports",
        )

    def test_system_has_three_voice_presentations_and_multiverse(self) -> None:
        self.assertEqual(validate_creative_variation(self.config), [])
        presentations = {
            voice["presentation"] for voice in self.config["voice_cast"]
        }
        self.assertEqual(
            presentations,
            {"neutral_robot", "masculine", "feminine"},
        )
        self.assertGreaterEqual(len(self.config["worlds"]), 6)

    def test_first_video_starts_with_robot(self) -> None:
        profile = select_creative_profile(self.candidate)
        self.assertEqual(profile["voice_id"], "robot_scout")
        self.assertEqual(validate_creative_profile(profile), [])

    def test_next_video_cannot_repeat_profile_dimensions(self) -> None:
        first = select_creative_profile(self.candidate)
        second = select_creative_profile(
            Candidate(
                candidate_id="story-2",
                title="Segundo cambio",
                source_url="https://example.com/story-2",
                territory="gaming_esports",
            ),
            history=[first],
        )
        self.assertNotEqual(first["voice_id"], second["voice_id"])
        self.assertNotEqual(first["world_id"], second["world_id"])
        self.assertNotEqual(first["hook_pattern"], second["hook_pattern"])
        self.assertNotEqual(first["motion_system"], second["motion_system"])

    def test_selection_is_stable_for_same_story_and_history(self) -> None:
        history = [
            {
                "voice_id": "robot_scout",
                "world_id": "cyber_arena",
                "hook_pattern": "desafio_directo",
                "motion_system": "circuitos_y_pulsos",
            }
        ]
        first = select_creative_profile(self.candidate, history)
        second = select_creative_profile(self.candidate, history)
        self.assertEqual(first, second)

    def test_real_person_voice_imitation_is_forbidden(self) -> None:
        profile = select_creative_profile(self.candidate)
        profile["invariants"] = [
            rule
            for rule in profile["invariants"]
            if rule != "no_imitar_personas_reales"
        ]
        self.assertIn(
            "missing_voice_identity_safety",
            validate_creative_profile(profile),
        )


if __name__ == "__main__":
    unittest.main()
