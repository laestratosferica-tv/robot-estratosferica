import unittest

from community_commercial_radar import (
    classify_interaction,
    prepare_outbound_action,
    prioritize_interactions,
)


class CommunityCommercialRadarTests(unittest.TestCase):
    def test_detects_hot_sponsorship_lead(self):
        result = classify_interaction({
            "text": "Nuestra marca quiere patrocinar un evento. ¿Cómo contrato una transmisión?",
            "display_name": "Marca X",
            "platform": "threads",
        })
        self.assertEqual(result["classification"], "commercial_lead")
        self.assertEqual(result["temperature"], "hot")
        self.assertEqual(result["intent"], "patrocinio")
        self.assertFalse(result["can_auto_send"])

    def test_keeps_normal_conversation_as_community(self):
        result = classify_interaction({
            "text": "Me gustó la noticia. ¿Cuál equipo creen que gana?",
            "platform": "threads",
        })
        self.assertEqual(result["classification"], "community_signal")
        self.assertEqual(result["temperature"], "cold")

    def test_marks_suspicious_interaction(self):
        result = classify_interaction({
            "text": "Ingreso garantizado, sígueme y te sigo",
            "platform": "instagram",
        })
        self.assertEqual(result["classification"], "risk_or_spam")
        self.assertEqual(result["reply_draft"], "")

    def test_contact_details_increase_score_without_exposing_or_sending(self):
        result = classify_interaction({
            "text": "Necesitamos una propuesta de contenido de marca. comercial@example.com",
            "platform": "threads",
        })
        self.assertEqual(result["classification"], "commercial_lead")
        self.assertIn("contacto_visible", result["reasons"])
        self.assertFalse(result["can_auto_send"])

    def test_prioritizes_commercial_leads(self):
        queue = prioritize_interactions([
            {"interaction_id": "community", "text": "Buen post"},
            {"interaction_id": "lead", "text": "Nuestra marca pide cotización de pauta"},
            {"interaction_id": "spam", "text": "Casino y dinero fácil"},
        ])
        self.assertEqual([item["interaction_id"] for item in queue], ["lead", "community", "spam"])

    def test_outbound_is_blocked_by_default(self):
        lead = classify_interaction({
            "interaction_id": "1",
            "text": "Nuestra empresa quiere una propuesta de patrocinio",
            "platform": "threads",
        })
        action = prepare_outbound_action(lead)
        self.assertEqual(action["status"], "blocked_pending_approval")
        self.assertFalse(action["sent"])
        self.assertEqual(action["text"], "")

    def test_approval_only_prepares_manual_send(self):
        lead = classify_interaction({
            "interaction_id": "1",
            "text": "Nuestra empresa quiere una propuesta de patrocinio",
            "platform": "threads",
        })
        action = prepare_outbound_action(
            lead, human_approved=True, allow_outbound=True
        )
        self.assertEqual(action["status"], "ready_for_manual_send")
        self.assertFalse(action["sent"])
        self.assertTrue(action["text"])


if __name__ == "__main__":
    unittest.main()
