from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum


class CEFRLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


# ─────────────────────────────────────────────
# INPUT: output langsung dari model i-Speak
# ─────────────────────────────────────────────

class KeyFeatures(BaseModel):
    """
    Key features yang diekstrak oleh i-Speak dari audio.
    Berdasarkan output real model i-Speak.
    """
    duration: float = Field(..., description="Durasi audio dalam detik")
    total_words: float = Field(..., description="Total kata yang diucapkan")
    wpm: float = Field(..., description="Words per minute (speech rate)")
    grammar_errors: int = Field(..., description="Jumlah grammar error terdeteksi")
    ttr: float = Field(..., ge=0, le=1, description="Type-Token Ratio (lexical diversity 0-1)")
    idioms_found: int = Field(..., description="Jumlah idiom yang ditemukan")
    cefr_distribution: Dict[str, int] = Field(
        ...,
        description="Distribusi kata per level CEFR. Contoh: {'A1': 40, 'B1': 4, 'C1': 0}"
    )


class SubConstructScore(BaseModel):
    """
    Skor per sub-construct dari model i-Speak.
    7 sub-constructs: Fluency, Pronunciation, Prosody,
    Coherence_and_Cohesion, Topic_Relevance, Complexity, Accuracy
    """
    name: str = Field(..., description="Nama sub-construct")
    cefr_level: CEFRLevel = Field(..., description="Level CEFR hasil prediksi model")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score model (0-1)")


class ISpeakOutput(BaseModel):
    """
    Seluruh output dari model i-Speak — input utama sistem kita.
    Menggabungkan KeyFeatures + 7 SubConstructScore.
    """
    key_features: KeyFeatures
    subconstructs: List[SubConstructScore] = Field(
        ...,
        description="List 7 sub-construct: Fluency, Pronunciation, Prosody, "
                    "Coherence_and_Cohesion, Topic_Relevance, Complexity, Accuracy"
    )
    task_type: str = Field(
        default="FreeSpeech",
        description="Tipe task i-Speak: FreeSpeech | ReadAloud | Interview"
    )

    @field_validator("subconstructs")
    @classmethod
    def validate_subconstructs(cls, v):
        valid_names = {
            "Fluency", "Pronunciation", "Prosody",
            "Coherence_and_Cohesion", "Topic_Relevance",
            "Complexity", "Accuracy"
        }
        for sc in v:
            if sc.name not in valid_names:
                raise ValueError(
                    f"Sub-construct '{sc.name}' tidak dikenal. "
                    f"Valid: {valid_names}"
                )
        return v

    def get_subconstruct(self, name: str) -> Optional[SubConstructScore]:
        """Helper: ambil subconstruct by name."""
        for sc in self.subconstructs:
            if sc.name == name:
                return sc
        return None

    def get_overall_cefr(self) -> CEFRLevel:
        """
        Hitung overall CEFR dari rata-rata weighted subconstructs.
        Weight: Fluency + Accuracy + Pronunciation masing-masing 20%,
                Complexity 15%, Prosody 10%, Coherence 10%, Topic 5%.
        """
        CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
        WEIGHTS = {
            "Fluency": 0.20,
            "Accuracy": 0.20,
            "Pronunciation": 0.20,
            "Complexity": 0.15,
            "Prosody": 0.10,
            "Coherence_and_Cohesion": 0.10,
            "Topic_Relevance": 0.05,
        }

        total_score = 0.0
        total_weight = 0.0

        for sc in self.subconstructs:
            w = WEIGHTS.get(sc.name, 0.1)
            idx = CEFR_ORDER.index(sc.cefr_level.value)
            total_score += idx * w
            total_weight += w

        if total_weight == 0:
            return CEFRLevel.A1

        avg_idx = round(total_score / total_weight)
        avg_idx = max(0, min(avg_idx, len(CEFR_ORDER) - 1))
        return CEFRLevel(CEFR_ORDER[avg_idx])


# ─────────────────────────────────────────────
# INTERNAL: profil diagnostik yang dibangun sistem
# ─────────────────────────────────────────────

class SubSkillLevel(BaseModel):
    """
    Level per sub-construct dengan metadata gap.
    
    Tambah confidence_moderation_note untuk transparansi
    perhitungan confidence-moderated weighted average.
    """
    sub_skill: str
    raw_score: float          
    cefr_level: CEFRLevel
    is_gap: bool              
    gap_magnitude: int        
    confidence_moderation_note: Optional[str] = Field(
        default=None,
        description="Catatan moderasi confidence, e.g. 'Confidence 30.1% -> bobot efektif 50%'"
    )


class FeatureInterpretation(BaseModel):
    """
    Feature-aware needs analysis.
    
    Menghubungkan KeyFeatures (WPM, grammar_errors, TTR, cefr_distribution)
    dengan kebutuhan pembelajaran spesifik.
    """
    indicator: str = Field(
        ...,
        description="Indikator kuantitatif, e.g. 'WPM=147 (terlalu lambat, normal B1=120-150)'"
    )
    learning_need: str = Field(
        ...,
        description="Kebutuhan pembelajaran, e.g. 'Latihan kelancaran berbicara dan pengurangan jeda panjang'"
    )
    drill_type: str = Field(
        ...,
        description="Latihan spesifik, e.g. 'Speed-building: read-aloud dengan timer, target 120 WPM'"
    )


class DiagnosticProfile(BaseModel):
    """
    Profil diagnostik terstruktur — grounding context untuk RAG & LLM.
    
    - confidence_moderation_applied: flag transparansi metode perhitungan
    - overall_cefr_calculation_method: dokumentasi metode yang digunakan
    - strength_skills: sub-construct di atas overall (untuk positive feedback)
    - feature_interpretations: interpretasi KeyFeatures per gap skill
    - priority_queries: max 8 query sesuai proposal
    """
    overall_cefr: CEFRLevel
    sub_skill_levels: List[SubSkillLevel]
    gap_skills: List[str]         
    strength_skills: List[str] = Field(
        default_factory=list,
        description="Sub-construct yang levelnya >= overall CEFR (untuk positive feedback)"
    )
    priority_queries: List[str] = Field(
        ...,
        max_length=8,
        description="Query semantik untuk RAG"
    )
    next_target_cefr: CEFRLevel    
    speech_rate_wpm: float
    ttr_score: float
    grammar_error_count: int
    dominant_cefr_vocab: str       
    
    confidence_moderation_applied: bool = Field(
        default=True,
        description="True jika confidence moderation diterapkan dalam perhitungan overall CEFR"
    )
    overall_cefr_calculation_method: str = Field(
        default="confidence_moderated_weighted_average",
        description="Metode perhitungan: 'confidence_moderated_weighted_average' atau 'simple_weighted_average'"
    )
    
    feature_interpretations: Dict[str, FeatureInterpretation] = Field(
        default_factory=dict,
        description="Interpretasi pedagogis KeyFeatures per gap skill. Key = nama sub-construct"
    )


class DiagnosticResponse(BaseModel):
    """Response sederhana untuk endpoint diagnostic saja."""
    overall_cefr: CEFRLevel
    gap_skills: List[str]
    queries: List[str]
    next_target: CEFRLevel


# ─────────────────────────────────────────────
# INTERNAL: hasil retrieval dari vector store
# ─────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    content: str
    source_doc: str
    doc_type: str
    cefr_level: str
    skill_focus: str
    similarity_score: float
    query_source: str


# ─────────────────────────────────────────────
# OUTPUT: feedback ke pengguna
# ─────────────────────────────────────────────

class SubConstructFeedback(BaseModel):
    """Feedback per sub-construct untuk ditampilkan ke user."""
    name: str
    cefr_level: CEFRLevel
    confidence: float
    is_gap: bool
    short_feedback: str    
    priority_exercise: str  


class CEFRAssessment(BaseModel):
    overall_level: CEFRLevel
    sub_construct_feedback: List[SubConstructFeedback]
    justification: str
    source_citations: List[str]


class ErrorCorrection(BaseModel):
    rank: int
    sub_skill: str
    current_level: CEFRLevel
    target_level: CEFRLevel
    explanation: str
    exercises: List[str]
    estimated_sessions: int
    source_citations: List[str]


class DailyTask(BaseModel):
    day: str
    activity_type: str
    description: str
    duration_minutes: int
    resource_reference: str


class WeeklyPlan(BaseModel):
    week: int
    focus: str
    daily_tasks: List[DailyTask]


class LearningPath(BaseModel):
    current_level: CEFRLevel
    target_level: CEFRLevel
    duration_weeks: int
    weekly_plan: List[WeeklyPlan]
    source_citations: List[str]


class FeedbackResponse(BaseModel):
    """
    Response utama yang dikembalikan ke pengguna.
    
    strategi prompt yang digunakan (sequential_4_prompt vs unified_1_prompt).
    """
    # Summary profil
    overall_cefr: CEFRLevel
    sub_construct_summary: List[SubConstructFeedback]
    # Detail assessment
    cefr_assessment: CEFRAssessment
    priority_corrections: List[ErrorCorrection]
    learning_path: LearningPath
    # Plain language
    plain_summary: str = Field(
        ..., description="Penjelasan dalam bahasa sederhana, max 100 kata"
    )
    top_priority_action: str
    # Metadata
    retrieval_sources: List[str]
    hallucination_check_passed: bool
    model_used: str
    generation_method: str = Field(
        default="sequential_4_prompt",
        description="Metode generasi: 'sequential_4_prompt' (4 step) atau 'unified_1_prompt' (1 step)"
    )


# ─────────────────────────────────────────────
# API response wrapper
# ─────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None