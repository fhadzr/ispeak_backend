"""
Diagnostic Service — CONFIDENCE-MODERATED VERSION
====================================================
Mengkonversi output i-Speak menjadi DiagnosticProfile 
dengan confidence moderation pada perhitungan overall CEFR.
"""

from app.models.schemas import (
    ISpeakOutput, DiagnosticProfile, SubSkillLevel, CEFRLevel,
    FeatureInterpretation
)

# ─────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]

ALL_SUBCONSTRUCTS = [
    "Fluency", "Pronunciation", "Prosody",
    "Coherence_and_Cohesion", "Topic_Relevance",
    "Complexity", "Accuracy",
]

# Bobot pedagogis sesuai Tabel 3.5 proposal
SUBCONSTRUCT_WEIGHTS = {
    "Fluency": 0.20,
    "Accuracy": 0.20,
    "Pronunciation": 0.20,
    "Complexity": 0.15,
    "Prosody": 0.10,
    "Coherence_and_Cohesion": 0.10,
    "Topic_Relevance": 0.05,
}

# Threshold confidence untuk moderasi
CONFIDENCE_MODERATION = {
    "low": (0.0, 0.50, 0.50),
    "moderate": (0.50, 0.70, 0.75),
    "high": (0.70, 1.0, 1.00),
}


def get_confidence_multiplier(confidence: float) -> float:
    """Hitung multiplier bobot berdasarkan confidence score."""
    for level, (min_conf, max_conf, multiplier) in CONFIDENCE_MODERATION.items():
        if min_conf <= confidence <= max_conf:
            return multiplier
    return 0.50


def calculate_confidence_moderated_cefr(subconstructs: list) -> CEFRLevel:
    """
    Hitung overall CEFR dengan confidence moderation.
    
    Formula: weighted_sum = Σ (cefr_index_i × base_weight_i × confidence_multiplier_i)
    Normalisasi: overall_index = round(weighted_sum / total_effective_weight)
    """
    total_effective_weight = 0.0
    weighted_sum = 0.0

    for sc in subconstructs:
        cefr_idx = CEFR_ORDER.index(sc.cefr_level.value)
        base_weight = SUBCONSTRUCT_WEIGHTS.get(sc.name, 0.10)
        conf_mult = get_confidence_multiplier(sc.confidence)
        effective_weight = base_weight * conf_mult
        
        total_effective_weight += effective_weight
        weighted_sum += cefr_idx * effective_weight

    if total_effective_weight == 0:
        return CEFRLevel("A1")

    normalized_index = weighted_sum / total_effective_weight
    rounded_index = round(normalized_index)
    rounded_index = max(0, min(rounded_index, len(CEFR_ORDER) - 1))
    
    return CEFRLevel(CEFR_ORDER[rounded_index])


# ─────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────

def build_queries_for_gap(sub_construct: str, current_level: str, overall_level: str) -> list[str]:
    templates = {
        "Fluency": [
            f"speaking fluency improvement {current_level} CEFR Indonesian learner exercises",
            f"silent pause reduction connected speech {current_level} to {overall_level}",
        ],
        "Pronunciation": [
            f"English pronunciation {current_level} Indonesian speaker remediation phoneme",
            f"Indonesian accent pronunciation improvement {current_level} CEFR exercises",
        ],
        "Prosody": [
            f"English intonation stress pattern {current_level} Indonesian speaker practice",
            f"prosody improvement {current_level} CEFR word stress rhythm exercises",
        ],
        "Coherence_and_Cohesion": [
            f"discourse coherence cohesion {current_level} CEFR speaking exercises",
            f"spoken discourse organization {current_level} to {overall_level} practice",
        ],
        "Topic_Relevance": [
            f"topic relevance vocabulary {current_level} CEFR speaking practice",
            f"on-topic speaking exercises {current_level} academic vocabulary",
        ],
        "Complexity": [
            f"lexical complexity vocabulary expansion {current_level} to {overall_level} CEFR",
            f"advanced vocabulary {current_level} CEFR word list exercises Academic Word List",
        ],
        "Accuracy": [
            f"grammar error correction {current_level} CEFR Indonesian speaker exercises",
            f"grammatical accuracy improvement {current_level} to {overall_level} speaking",
        ],
    }
    return templates.get(sub_construct, [
        f"{sub_construct} improvement {current_level} CEFR speaking exercises"
    ])


def get_dominant_cefr_vocab(cefr_distribution: dict) -> str:
    if not cefr_distribution:
        return "A1"
    return max(cefr_distribution, key=lambda k: cefr_distribution[k])


# ─────────────────────────────────────────────
# Feature-Aware Interpretation (Langkah 5)
# ─────────────────────────────────────────────

def interpret_key_features(key_features, gap_skills: list[str]) -> dict[str, FeatureInterpretation]:
    """
    Interpretasi pedagogis berbasis KeyFeatures (Tabel 3.6 proposal).
    """
    interpretations = {}
    
    # WPM → Fluency
    wpm = key_features.wpm
    if "Fluency" in gap_skills:
        if wpm < 100:
            interpretations["Fluency"] = FeatureInterpretation(
                indicator=f"WPM={wpm:.0f} (terlalu lambat, normal B1=120-150)",
                learning_need="Latihan kelancaran berbicara dan pengurangan jeda panjang",
                drill_type="Speed-building: read-aloud dengan timer, target 120 WPM"
            )
        elif wpm > 180:
            interpretations["Fluency"] = FeatureInterpretation(
                indicator=f"WPM={wpm:.0f} (terlalu cepat, berpotensi unclear)",
                learning_need="Latihan articulation clarity pada kecepatan tinggi",
                drill_type="Shadowing native speaker dengan fokus clarity"
            )
        else:
            interpretations["Fluency"] = FeatureInterpretation(
                indicator=f"WPM={wpm:.0f} (dalam rentang normal)",
                learning_need="Fokus pada pause pattern dan connected speech",
                drill_type="Connected speech drills: link words, reduce weak forms"
            )
    
    # Grammar Errors → Accuracy
    grammar_errors = key_features.grammar_errors
    if "Accuracy" in gap_skills:
        if grammar_errors >= 5:
            interpretations["Accuracy"] = FeatureInterpretation(
                indicator=f"Grammar errors={grammar_errors} (tinggi, masalah fundamental)",
                learning_need="Remediasi pola kesalahan tata bahasa spesifik, khususnya L1 Indonesia transfer",
                drill_type="Pattern drilling: Indonesian-English contrastive analysis exercises"
            )
        elif grammar_errors >= 2:
            interpretations["Accuracy"] = FeatureInterpretation(
                indicator=f"Grammar errors={grammar_errors} (moderate, pola tertentu)",
                learning_need="Identifikasi dan koreksi pola error berulang",
                drill_type="Error correction log: catat, klasifikasi, dan drill pola spesifik"
            )
        else:
            interpretations["Accuracy"] = FeatureInterpretation(
                indicator=f"Grammar errors={grammar_errors} (rendah, sporadis)",
                learning_need="Konsolidasi accuracy untuk complex structures",
                drill_type="Advanced grammar: conditional, passive, reported speech"
            )
    
    # TTR → Complexity
    ttr = key_features.ttr
    if "Complexity" in gap_skills:
        if ttr < 0.40:
            interpretations["Complexity"] = FeatureInterpretation(
                indicator=f"TTR={ttr:.2f} (sangat rendah, kosakata repetitif)",
                learning_need="Pengayaan vocabulary dan latihan variasi ekspresi",
                drill_type="Synonym substitution: ganti 5 kata umum dengan alternatif level target"
            )
        elif ttr < 0.60:
            interpretations["Complexity"] = FeatureInterpretation(
                indicator=f"TTR={ttr:.2f} (rendah, perlu diversifikasi)",
                learning_need="Latihan lexical diversity dan academic vocabulary",
                drill_type="Oxford 3000/5000 drills: pelajari 10 kata baru per hari sesuai level"
            )
        else:
            interpretations["Complexity"] = FeatureInterpretation(
                indicator=f"TTR={ttr:.2f} (cukup baik, fokus sophisticated vocabulary)",
                learning_need="Peningkatan ke lexical sophistication (idioms, collocations)",
                drill_type="Academic Word List + collocation matching exercises"
            )
    
    # CEFR Distribution → Topic Relevance
    cefr_dist = key_features.cefr_distribution
    if "Topic_Relevance" in gap_skills:
        dominant = get_dominant_cefr_vocab(cefr_dist)
        if dominant in ["A1", "A2"]:
            interpretations["Topic_Relevance"] = FeatureInterpretation(
                indicator=f"Dominant vocab: {dominant} ({cefr_dist.get(dominant, 0)} kata) — terlalu dasar",
                learning_need="Topic-focused vocabulary expansion ke level target",
                drill_type="Topic mind-mapping: brainstorm 20 kata level B1 untuk topik spesifik"
            )
        else:
            interpretations["Topic_Relevance"] = FeatureInterpretation(
                indicator=f"Dominant vocab: {dominant} — cukup baik, perlu relevansi konteks",
                learning_need="Latihan on-topic speaking dengan academic vocabulary",
                drill_type="Timed speaking: 2 menit tentang topik dengan minimal 5 target words"
            )
    
    return interpretations


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def build_diagnostic_profile(ispeak_output: ISpeakOutput) -> DiagnosticProfile:
    """
    Konversi output i-Speak → DiagnosticProfile dengan:
    1. Confidence moderation (Langkah 3)
    2. Feature-aware interpretation (Langkah 5)
    3. Max 8 queries (Langkah 1 — sudah Anda lakukan)
    """
    
    # Step 1: Hitung overall CEFR dengan CONFIDENCE MODERATION
    overall_cefr = calculate_confidence_moderated_cefr(ispeak_output.subconstructs)
    overall_idx = CEFR_ORDER.index(overall_cefr.value)
    
    # Step 2 & 3: Map subconstructs, deteksi gap
    sub_skill_levels = []
    gap_skills = []
    strength_skills = []
    all_queries = []
    
    for sc in ispeak_output.subconstructs:
        sc_idx = CEFR_ORDER.index(sc.cefr_level.value)
        is_gap = sc_idx < overall_idx
        gap_magnitude = max(0, overall_idx - sc_idx)
        
        # Confidence moderation note
        conf_mult = get_confidence_multiplier(sc.confidence)
        conf_note = None
        if conf_mult < 1.0:
            conf_note = f"Confidence {sc.confidence:.1%} → bobot efektif {conf_mult:.0%}"
        
        sub_skill_levels.append(SubSkillLevel(
            sub_skill=sc.name,
            raw_score=sc.confidence,
            cefr_level=sc.cefr_level,
            is_gap=is_gap,
            gap_magnitude=gap_magnitude,
            confidence_moderation_note=conf_note,
        ))
        
        if is_gap:
            gap_skills.append(sc.name)
            queries = build_queries_for_gap(sc.name, sc.cefr_level.value, overall_cefr.value)
            all_queries.extend(queries)
        else:
            strength_skills.append(sc.name)
    
    # Query umum jika tidak ada gap
    if not all_queries:
        all_queries = [
            f"CEFR {overall_cefr.value} speaking improvement exercises general",
            f"English speaking practice {overall_cefr.value} level learning path",
        ]
    
    # Deduplicate + max 8 queries
    seen = set()
    unique_queries = []
    for q in all_queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
        if len(unique_queries) >= 8:
            break
    
    # Target level
    next_idx = min(overall_idx + 1, len(CEFR_ORDER) - 1)
    next_target = CEFRLevel(CEFR_ORDER[next_idx])
    
    # Dominant vocab
    dominant_vocab = get_dominant_cefr_vocab(
        ispeak_output.key_features.cefr_distribution
    )
    
    # Step 5: Feature-aware interpretation
    feature_interpretations = interpret_key_features(
        ispeak_output.key_features,
        gap_skills,
    )
    
    return DiagnosticProfile(
        overall_cefr=overall_cefr,
        sub_skill_levels=sub_skill_levels,
        gap_skills=gap_skills,
        strength_skills=strength_skills,
        priority_queries=unique_queries,
        next_target_cefr=next_target,
        speech_rate_wpm=ispeak_output.key_features.wpm,
        ttr_score=ispeak_output.key_features.ttr,
        grammar_error_count=ispeak_output.key_features.grammar_errors,
        dominant_cefr_vocab=dominant_vocab,
        feature_interpretations=feature_interpretations,
        confidence_moderation_applied=True,
        overall_cefr_calculation_method="confidence_moderated_weighted_average",
    )