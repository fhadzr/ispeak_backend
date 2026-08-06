"""
Integration test untuk API endpoint POST /api/v1/assess.
Menggunakan mock untuk Supabase dan LLM agar tidak perlu koneksi nyata.
Jalankan: pytest tests/test_api.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ─────────────────────────────────────────────
# Payload contoh — sesuai output real i-Speak
# ─────────────────────────────────────────────

VALID_PAYLOAD = {
    "key_features": {
        "duration": 41.98,
        "total_words": 103,
        "wpm": 147.23,
        "grammar_errors": 2,
        "ttr": 0.64,
        "idioms_found": 1,
        "cefr_distribution": {"A1": 40, "B1": 4, "C1": 0}
    },
    "subconstructs": [
        {"name": "Fluency",               "cefr_level": "B2", "confidence": 0.963},
        {"name": "Pronunciation",          "cefr_level": "B1", "confidence": 0.996},
        {"name": "Prosody",               "cefr_level": "A2", "confidence": 0.899},
        {"name": "Coherence_and_Cohesion","cefr_level": "A2", "confidence": 0.764},
        {"name": "Topic_Relevance",        "cefr_level": "A1", "confidence": 0.712},
        {"name": "Complexity",            "cefr_level": "B1", "confidence": 1.000},
        {"name": "Accuracy",              "cefr_level": "C2", "confidence": 0.301},
    ],
    "task_type": "FreeSpeech"
}

INVALID_PAYLOAD_BAD_SUBCONSTRUCT = {
    "key_features": {
        "duration": 30.0, "total_words": 80, "wpm": 120.0,
        "grammar_errors": 3, "ttr": 0.5, "idioms_found": 0,
        "cefr_distribution": {"A1": 20}
    },
    "subconstructs": [
        {"name": "InvalidSubconstruct", "cefr_level": "B1", "confidence": 0.8},
    ],
    "task_type": "FreeSpeech"
}


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestAssessEndpoint:

    def test_invalid_subconstruct_rejected(self):
        """Payload dengan subconstruct name tidak valid harus 422."""
        response = client.post("/api/v1/assess", json=INVALID_PAYLOAD_BAD_SUBCONSTRUCT)
        assert response.status_code == 422

    def test_missing_key_features_rejected(self):
        """Payload tanpa key_features harus 422."""
        payload = {"subconstructs": VALID_PAYLOAD["subconstructs"]}
        response = client.post("/api/v1/assess", json=payload)
        assert response.status_code == 422

    def test_missing_subconstructs_rejected(self):
        """Payload tanpa subconstructs harus 422."""
        payload = {"key_features": VALID_PAYLOAD["key_features"]}
        response = client.post("/api/v1/assess", json=payload)
        assert response.status_code == 422

    @patch("app.api.v1.routes.assess.retrieve_all")
    @patch("app.api.v1.routes.assess.run_full_diagnosis")
    def test_valid_payload_returns_success(self, mock_llm, mock_retrieve):
        """Payload valid dengan mock LLM dan mock retrieval harus return 200."""
        from app.models.schemas import (
            FeedbackResponse, CEFRAssessment, LearningPath,
            CEFRLevel, SubConstructFeedback
        )

        # Mock retrieval kosong (knowledge base belum ada)
        mock_retrieve.return_value = []

        # Mock LLM response
        mock_llm.return_value = FeedbackResponse(
            overall_cefr=CEFRLevel.B1,
            sub_construct_summary=[
                SubConstructFeedback(
                    name="Fluency", cefr_level=CEFRLevel.B2,
                    confidence=0.963, is_gap=False,
                    short_feedback="Fluency baik di B2",
                    priority_exercise="Continue shadowing practice",
                )
            ],
            cefr_assessment=CEFRAssessment(
                overall_level=CEFRLevel.B1,
                sub_construct_feedback=[],
                justification="Based on CEFR descriptors",
                source_citations=["cefr_descriptors.md"],
            ),
            priority_corrections=[],
            learning_path=LearningPath(
                current_level=CEFRLevel.B1,
                target_level=CEFRLevel.B2,
                duration_weeks=2,
                weekly_plan=[],
                source_citations=[],
            ),
            plain_summary="Kemampuan berbicara Anda secara keseluruhan berada di level B1.",
            top_priority_action="Latih Topic_Relevance dan Prosody setiap hari.",
            retrieval_sources=[],
            hallucination_check_passed=True,
            model_used="qwen2.5-72b-instruct",
        )

        response = client.post("/api/v1/assess", json=VALID_PAYLOAD)
        assert response.status_code == 200

        body = response.json()
        assert body["success"] is True
        assert body["data"] is not None
        assert body["data"]["overall_cefr"] == "B1"

    def test_ttr_out_of_range_rejected(self):
        """TTR > 1 harus ditolak validasi."""
        bad = dict(VALID_PAYLOAD)
        bad["key_features"] = dict(VALID_PAYLOAD["key_features"])
        bad["key_features"]["ttr"] = 1.5  # tidak valid
        response = client.post("/api/v1/assess", json=bad)
        assert response.status_code == 422