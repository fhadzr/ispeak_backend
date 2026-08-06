"""
Unit tests untuk diagnostic service.
Test ini tidak memerlukan koneksi Supabase atau LLM.
Jalankan: pytest tests/test_diagnostic.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schemas import (
    ISpeakOutput, KeyFeatures, SubConstructScore, CEFRLevel
)
from app.services.diagnostic import build_diagnostic_profile


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_ispeak_output(subconstructs: dict) -> ISpeakOutput:
    """
    Helper: buat ISpeakOutput dari dict {name: (cefr_level, confidence)}.
    Contoh dari screenshot:
      Fluency: B2 96.3%, Pronunciation: B1 99.6%, Prosody: A2 89.9%
      Coherence_and_Cohesion: A2 76.4%, Topic_Relevance: A1 71.2%
      Complexity: B1 100%, Accuracy: C2 30.1%
    """
    return ISpeakOutput(
        key_features=KeyFeatures(
            duration=41.98,
            total_words=103,
            wpm=147.23,
            grammar_errors=2,
            ttr=0.64,
            idioms_found=1,
            cefr_distribution={"A1": 40, "B1": 4, "C1": 0},
        ),
        subconstructs=[
            SubConstructScore(name=name, cefr_level=CEFRLevel(level), confidence=conf)
            for name, (level, conf) in subconstructs.items()
        ],
        task_type="FreeSpeech",
    )


# Output real dari screenshot i-Speak
SCREENSHOT_SUBCONSTRUCTS = {
    "Fluency":               ("B2", 0.963),
    "Pronunciation":         ("B1", 0.996),
    "Prosody":               ("A2", 0.899),
    "Coherence_and_Cohesion":("A2", 0.764),
    "Topic_Relevance":       ("A1", 0.712),
    "Complexity":            ("B1", 1.000),
    "Accuracy":              ("C2", 0.301),
}

# Semua subconstruct di level yang sama (tidak ada gap)
NO_GAP_SUBCONSTRUCTS = {
    "Fluency":               ("B1", 0.900),
    "Pronunciation":         ("B1", 0.850),
    "Prosody":               ("B1", 0.800),
    "Coherence_and_Cohesion":("B1", 0.820),
    "Topic_Relevance":       ("B1", 0.780),
    "Complexity":            ("B1", 0.900),
    "Accuracy":              ("B1", 0.870),
}


class TestBuildDiagnosticProfile:

    def test_overall_cefr_calculated(self):
        """Overall CEFR harus dihitung dari weighted average, bukan dari input."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        # Dengan data screenshot, overall seharusnya B1 atau B2
        assert profile.overall_cefr in [CEFRLevel.B1, CEFRLevel.B2]

    def test_gap_skills_detected(self):
        """Prosody, Coherence, Topic_Relevance harus terdeteksi sebagai gap."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        # Topic_Relevance A1 pasti gap
        assert "Topic_Relevance" in profile.gap_skills
        # Prosody A2 pasti gap jika overall >= B1
        assert "Prosody" in profile.gap_skills

    def test_no_gap_when_all_same_level(self):
        """Jika semua subconstruct sama level, tidak ada gap."""
        output = make_ispeak_output(NO_GAP_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        assert len(profile.gap_skills) == 0

    def test_priority_queries_generated(self):
        """Harus ada priority queries yang dihasilkan."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        assert len(profile.priority_queries) > 0
        assert len(profile.priority_queries) <= 10

    def test_priority_queries_max_when_no_gap(self):
        """Tanpa gap, tetap ada query default (bukan list kosong)."""
        output = make_ispeak_output(NO_GAP_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        assert len(profile.priority_queries) >= 1

    def test_next_target_one_level_up(self):
        """next_target_cefr harus satu level di atas overall."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
        overall_idx = CEFR_ORDER.index(profile.overall_cefr.value)
        target_idx = CEFR_ORDER.index(profile.next_target_cefr.value)
        assert target_idx == min(overall_idx + 1, 5)

    def test_sub_skill_levels_count(self):
        """Harus ada 7 SubSkillLevel — satu per subconstruct."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        assert len(profile.sub_skill_levels) == 7

    def test_key_features_in_profile(self):
        """key_features harus masuk ke profile untuk konteks LLM."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        assert profile.speech_rate_wpm == 147.23
        assert profile.ttr_score == 0.64
        assert profile.grammar_error_count == 2

    def test_dominant_cefr_vocab(self):
        """dominant_cefr_vocab harus A1 karena A1=40 adalah terbanyak."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        assert profile.dominant_cefr_vocab == "A1"

    def test_gap_magnitude_correct(self):
        """Magnitude gap Topic_Relevance A1 vs overall B1 harus >= 2."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        profile = build_diagnostic_profile(output)
        topic_sl = next(
            (sl for sl in profile.sub_skill_levels if sl.sub_skill == "Topic_Relevance"),
            None
        )
        assert topic_sl is not None
        assert topic_sl.gap_magnitude >= 2  # A1 ke B1 = 2 level


class TestISpeakOutputModel:

    def test_get_subconstruct_helper(self):
        """Helper get_subconstruct harus return subconstruct yang benar."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        fluency = output.get_subconstruct("Fluency")
        assert fluency is not None
        assert fluency.cefr_level == CEFRLevel.B2
        assert fluency.confidence == 0.963

    def test_get_subconstruct_not_found(self):
        """Helper harus return None jika tidak ditemukan."""
        output = make_ispeak_output(SCREENSHOT_SUBCONSTRUCTS)
        result = output.get_subconstruct("NonExistent")
        assert result is None

    def test_invalid_subconstruct_name_rejected(self):
        """Nama subconstruct yang tidak valid harus ditolak Pydantic."""
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ISpeakOutput(
                key_features=KeyFeatures(
                    duration=10, total_words=50, wpm=100,
                    grammar_errors=1, ttr=0.5, idioms_found=0,
                    cefr_distribution={"A1": 10},
                ),
                subconstructs=[
                    SubConstructScore(
                        name="InvalidName",  # bukan salah satu dari 7 valid
                        cefr_level=CEFRLevel.B1,
                        confidence=0.9,
                    )
                ],
                task_type="FreeSpeech",
            )