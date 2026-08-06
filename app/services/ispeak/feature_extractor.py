"""
app/services/ispeak/feature_extractor.py — FIXED

Fitur disesuaikan PERSIS dengan jumlah yang di-fit scaler SubModel_fixed.py:
  Fluency               : 71  (7 named + 64 derived)
  Pronunciation         : 35  (3 named + 32 derived)
  Prosody               : 71  (7 named + 64 derived)
  Coherence_and_Cohesion: 35  (3 named + 32 derived)
  Topic_Relevance       : 11  (1 named + 10 derived)
  Complexity            : 131 (17 named + 114 derived)
  Accuracy              : 11  (1 named + 10 derived)

Cara SubModel_fixed.py menghasilkan fitur:
  feature_engineering(X_raw, feats) → transformasi matematis dari tabular features
  Bukan audio features (librosa) — murni dari kolom CSV

Karena serving pakai audio, kita:
1. Ekstrak named features dari audio (sama seperti sebelumnya)
2. Generate derived features secara MATEMATIS dari named features
   (log, sqrt, square, sin, cos, stats, interactions) — sama persis
   dengan ultra_feature_engineering() di SubModel_fixed.py
3. Pad/trim ke target size
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — replikasi SubModel_fixed.py feature_engineering()
# ═══════════════════════════════════════════════════════════════════════════

def _math_feature_engineering(named: np.ndarray, target_size: int) -> np.ndarray:
    """
    Replikasi feature_engineering() dari SubModel_fixed.py.
    Input: array 1D named features
    Output: array 1D dengan panjang = target_size
    
    Urutan transformasi (sama persis dengan SubModel_fixed.py):
    1. Original features
    2. log1p(|x|)
    3. sqrt(|x|)
    4. x^2
    5. sin(x)
    6. cos(x)
    7. Statistical: mean, std, max, min, median (5 values)
    8. Polynomial interactions (max 4 fitur pertama): x*y, x/y, x+y, x-y
    """
    X = named.reshape(1, -1).astype(np.float64)
    X_abs = np.abs(X) + 1e-6

    features = [
        X,                    # original
        np.log1p(X_abs),      # log
        np.sqrt(X_abs),       # sqrt
        X ** 2,               # square
        np.sin(X),            # sin
        np.cos(X),            # cos
    ]

    # Statistical features (5)
    stats = np.array([[
        np.mean(X),
        np.std(X),
        np.max(X),
        np.min(X),
        np.median(X),
    ]])
    features.append(stats)

    # Polynomial interactions (max 4 fitur pertama)
    n_interact = min(len(named), 4)
    for i in range(n_interact):
        for j in range(i + 1, n_interact):
            ci = X[0, i]
            cj = X[0, j]
            interaction = np.array([[
                ci * cj,
                ci / (cj + 1e-6),
                ci + cj,
                ci - cj,
            ]])
            features.append(interaction)

    result = np.concatenate(features, axis=1).flatten()
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad atau trim ke target_size
    if len(result) < target_size:
        result = np.concatenate([result, np.zeros(target_size - len(result))])
    else:
        result = result[:target_size]

    return result.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# SCALER APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def apply_scaler(feature_vec: np.ndarray, scaler_tuple) -> np.ndarray:
    """
    Apply PowerTransformer lalu MinMaxScaler.
    Input: array 1D float64 dengan panjang = n_features_in_ scaler
    Output: array 1D float32
    """
    power_t, minmax, _ = scaler_tuple
    x = feature_vec.astype(np.float64).reshape(1, -1)
    x = power_t.transform(x)
    x = minmax.transform(x)
    return x.flatten().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path: str) -> str:
    """Transkripsi audio ke teks menggunakan OpenAI Whisper."""
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    return result["text"].strip()


# ═══════════════════════════════════════════════════════════════════════════
# FLUENCY — 71 fitur
# Named (7): Total Words, WPM, WPS, Filled Pauses, MLR, Pause Freq, Durasi
# ═══════════════════════════════════════════════════════════════════════════

FILLED_PAUSE_WORDS = {
    "um","uh","er","eh","ah","hmm","hm","umm","uhh","err",
    "erm","like","basically","literally","right","okay","so","well"
}

def extract_fluency_features(
    audio_path: str,
    transcript: str,
    scaler_tuple,
) -> np.ndarray:
    """Output: array (1, 1, 71) siap masuk model GRU Fluency."""
    import librosa

    y, sr = librosa.load(audio_path, sr=16000)
    duration = float(librosa.get_duration(y=y, sr=sr))
    words = transcript.strip().split()
    total_words = len(words)
    wpm  = (total_words / duration * 60) if duration > 0 else 0.0
    wps  = (total_words / duration)       if duration > 0 else 0.0

    fp_count = sum(1 for w in words if w.lower().strip(".,!?") in FILLED_PAUSE_WORDS)

    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    for i in range(1, len(intervals)):
        gap = (intervals[i][0] - intervals[i-1][1]) / sr
        if gap > 0:
            pauses.append(gap)
    long_pauses = [p for p in pauses if p > 0.5]
    pause_freq  = len(long_pauses)

    runs, cur = [], []
    for w in words:
        if w.lower().strip(".,!?") in FILLED_PAUSE_WORDS:
            if cur: runs.append(len(cur)); cur = []
        else:
            cur.append(w)
    if cur: runs.append(len(cur))
    mlr = float(np.mean(runs)) if runs else 0.0

    named = np.array([
        total_words, wpm, wps, fp_count, mlr, pause_freq, duration
    ], dtype=np.float64)

    feature_vec = _math_feature_engineering(named, target_size=71)
    feature_vec = apply_scaler(feature_vec, scaler_tuple)
    return feature_vec.reshape(1, 1, 71)


# ═══════════════════════════════════════════════════════════════════════════
# PRONUNCIATION — 35 fitur
# Named (3): Articulation Rate, Pitch Range (Hz), MFCC (%)
# ═══════════════════════════════════════════════════════════════════════════

def extract_pronunciation_features(
    audio_path: str,
    transcript: str,
    scaler_tuple,
) -> np.ndarray:
    """Output: array (1, 1, 35) siap masuk model GRU Pronunciation."""
    import librosa

    y, sr = librosa.load(audio_path, sr=16000)
    duration = float(librosa.get_duration(y=y, sr=sr))
    words    = transcript.strip().split()

    intervals  = librosa.effects.split(y, top_db=30)
    speech_dur = sum((e - s) for s, e in intervals) / sr
    ar = (len(words) / speech_dur) if speech_dur > 0 else 0.0

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes[magnitudes > 0]) if magnitudes[magnitudes > 0].size > 0 else [0.01]]
    pitch_vals = pitch_vals[pitch_vals > 50]
    pitch_min  = float(np.min(pitch_vals))  if len(pitch_vals) > 0 else 0.0
    pitch_max  = float(np.max(pitch_vals))  if len(pitch_vals) > 0 else 0.0
    pitch_range = pitch_max - pitch_min

    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    half    = mfcc.shape[1] // 2
    if half > 0:
        m1   = np.mean(mfcc[:, :half], axis=1)
        m2   = np.mean(mfcc[:, half:], axis=1)
        norm = np.linalg.norm(m1) * np.linalg.norm(m2)
        mfcc_cos = float(np.dot(m1, m2) / norm) if norm > 0 else 0.5
    else:
        mfcc_cos = 0.5

    named = np.array([ar, pitch_range, mfcc_cos * 100], dtype=np.float64)

    feature_vec = _math_feature_engineering(named, target_size=35)
    feature_vec = apply_scaler(feature_vec, scaler_tuple)
    return feature_vec.reshape(1, 1, 35)


# ═══════════════════════════════════════════════════════════════════════════
# PROSODY — 71 fitur
# Named (7): Mean Pitch, Stdev Pitch, Mean Energy, Stdev Energy,
#            Num Prominences, Prominence Dist Mean, Prominence Dist Std
# ═══════════════════════════════════════════════════════════════════════════

def extract_prosody_features(
    audio_path: str,
    scaler_tuple,
) -> np.ndarray:
    """Output: array (1, 1, 71) siap masuk model GRU Prosody."""
    import librosa
    from scipy.signal import find_peaks

    y, sr = librosa.load(audio_path, sr=16000)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes[magnitudes > 0]) if magnitudes[magnitudes > 0].size > 0 else [0.01]]
    pitch_vals = pitch_vals[pitch_vals > 50]
    pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    pitch_std  = float(np.std(pitch_vals))  if len(pitch_vals) > 0 else 0.0

    rms         = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    peaks, _  = find_peaks(rms, height=np.mean(rms), distance=10)
    num_prom  = float(len(peaks))
    if len(peaks) > 1:
        dists = np.diff(peaks.astype(float))
        prom_dist_mean = float(np.mean(dists))
        prom_dist_std  = float(np.std(dists))
    else:
        prom_dist_mean = 0.0
        prom_dist_std  = 0.0

    named = np.array([
        pitch_mean, pitch_std,
        energy_mean, energy_std,
        num_prom, prom_dist_mean, prom_dist_std,
    ], dtype=np.float64)

    feature_vec = _math_feature_engineering(named, target_size=71)
    feature_vec = apply_scaler(feature_vec, scaler_tuple)
    return feature_vec.reshape(1, 1, 71)


# ═══════════════════════════════════════════════════════════════════════════
# COHERENCE & COHESION — 35 fitur
# Named (3): Semantic Coherence (%), Discourse Count, Linking Count
# ═══════════════════════════════════════════════════════════════════════════

DISCOURSE_MARKERS = {
    "however","therefore","furthermore","moreover","nevertheless",
    "consequently","thus","hence","although","whereas","meanwhile",
    "additionally","in contrast","on the other hand","as a result",
    "in conclusion","for example","for instance","in addition",
    "in summary","to sum up","in other words","that said"
}
LINKING_WORDS = {
    "and","but","so","because","although","if","when","while","since",
    "unless","until","before","after","as","that","which","who",
    "however","therefore","moreover","furthermore","also","yet","or","nor"
}

def extract_coherence_features(
    transcript: str,
    scaler_tuple,
) -> np.ndarray:
    """Output: array (1, 1, 35) siap masuk model GRU Coherence."""
    import re
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
    words     = transcript.lower().split()

    dm_count = sum(1 for m in DISCOURSE_MARKERS if m in transcript.lower())
    lw_count = sum(1 for w in words if w.strip(".,!?;:'\"") in LINKING_WORDS)

    try:
        sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        if len(sentences) >= 2:
            embs     = sbert.encode(sentences)
            sims     = [float(cos_sim([embs[i]], [embs[i+1]])[0][0])
                        for i in range(len(embs)-1)]
            coh_mean = float(np.mean(sims)) * 100
        else:
            coh_mean = 50.0
    except Exception:
        coh_mean = 50.0

    named = np.array([coh_mean, float(dm_count), float(lw_count)], dtype=np.float64)

    feature_vec = _math_feature_engineering(named, target_size=35)
    feature_vec = apply_scaler(feature_vec, scaler_tuple)
    return feature_vec.reshape(1, 1, 35)


# ═══════════════════════════════════════════════════════════════════════════
# TOPIC RELEVANCE — 11 fitur
# Named (1): Topic Similarity (%)
# ═══════════════════════════════════════════════════════════════════════════

def extract_topic_relevance_features(
    transcript: str,
    scaler_tuple,
    prompt_text: str = "",
) -> np.ndarray:
    """Output: array (1, 1, 11) siap masuk model GRU Topic Relevance."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    try:
        sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        emb_t = sbert.encode([transcript])[0]
        if prompt_text.strip():
            emb_p     = sbert.encode([prompt_text])[0]
            topic_sim = float(cos_sim([emb_t], [emb_p])[0][0]) * 100
        else:
            topic_sim = 50.0
    except Exception:
        topic_sim = 50.0

    named = np.array([topic_sim], dtype=np.float64)

    feature_vec = _math_feature_engineering(named, target_size=11)
    feature_vec = apply_scaler(feature_vec, scaler_tuple)
    return feature_vec.reshape(1, 1, 11)


# ═══════════════════════════════════════════════════════════════════════════
# COMPLEXITY — 131 fitur
# Named (17): Idioms Found, Bigram-Fourgram Count, Synonym Variations,
#             CEFR A1-C2 UNKNOWN, Avg/Max Tree Depth,
#             Token Count, Type Count, TTR
# ═══════════════════════════════════════════════════════════════════════════

IDIOM_LIST = {
    "on the other hand","in the long run","at the end of the day",
    "break a leg","cost an arm and a leg","hit the nail on the head",
    "once in a blue moon","under the weather","bite the bullet",
    "add fuel to the fire","burn the midnight oil","piece of cake",
    "spill the beans","let the cat out of the bag","beat around the bush"
}
BASIC_WORDS = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","can","could","shall","should","may",
    "might","must","i","you","he","she","it","we","they","my","your","his",
    "her","its","our","their","this","that","these","those","of","in","to",
    "for","on","with","at","by","from","up","about","into","through","than"
}

def extract_complexity_features(
    transcript: str,
    scaler_tuple,
) -> np.ndarray:
    """Output: array (1, 1, 131) siap masuk model GRU Complexity."""
    from collections import Counter

    words  = [w.lower().strip(".,!?;:'\"") for w in transcript.split() if w.strip()]
    alpha  = [w for w in words if w.isalpha()]
    total  = len(alpha)
    types  = len(set(alpha))
    ttr    = types / max(total, 1)

    bigrams   = list(zip(alpha[:-1], alpha[1:]))
    trigrams  = list(zip(alpha[:-2], alpha[1:-1], alpha[2:]))
    fourgrams = list(zip(alpha[:-3], alpha[1:-2], alpha[2:-1], alpha[3:]))

    text_l      = transcript.lower()
    idioms_found = sum(1 for idiom in IDIOM_LIST if idiom in text_l)

    freq    = Counter(alpha)
    hapax   = sum(1 for c in freq.values() if c == 1)
    syn_var = hapax

    basic   = sum(1 for w in alpha if w in BASIC_WORDS)
    adv_w   = [w for w in alpha if w not in BASIC_WORDS]
    cefr_a1 = basic
    cefr_a2 = sum(1 for w in adv_w if len(w) <= 5)
    cefr_b1 = sum(1 for w in adv_w if 5 < len(w) <= 7)
    cefr_b2 = sum(1 for w in adv_w if 7 < len(w) <= 9)
    cefr_c1 = sum(1 for w in adv_w if 9 < len(w) <= 11)
    cefr_c2 = sum(1 for w in adv_w if len(w) > 11)
    cefr_unk = 0

    sents     = [s.strip() for s in transcript.split('.') if s.strip()]
    sent_lens = [len(s.split()) for s in sents]
    avg_tree  = float(np.mean(sent_lens)) if sent_lens else 0.0
    max_tree  = float(np.max(sent_lens))  if sent_lens else 0.0

    named = np.array([
        float(idioms_found),
        float(len(bigrams)),
        float(len(trigrams)),
        float(len(fourgrams)),
        float(syn_var),
        float(cefr_a1), float(cefr_a2), float(cefr_b1), float(cefr_b2),
        float(cefr_c1), float(cefr_c2), float(cefr_unk),
        avg_tree, max_tree,
        float(total), float(types), ttr,
    ], dtype=np.float64)

    feature_vec = _math_feature_engineering(named, target_size=131)
    feature_vec = apply_scaler(feature_vec, scaler_tuple)
    return feature_vec.reshape(1, 1, 131)


# ═══════════════════════════════════════════════════════════════════════════
# ACCURACY — 11 fitur
# Named (1): Grammar Errors
# ═══════════════════════════════════════════════════════════════════════════

def extract_accuracy_features(
    audio_path: str,
    transcript: str,
    scaler_tuple,
) -> np.ndarray:
    """Output: array (1, 1, 11) siap masuk model GRU Accuracy."""
    words = transcript.strip().split()

    try:
        import language_tool_python
        tool           = language_tool_python.LanguageTool('en-US')
        grammar_errors = len(tool.check(transcript))
        tool.close()
    except Exception:
        grammar_errors = 0

    named = np.array([float(grammar_errors)], dtype=np.float64)

    feature_vec = _math_feature_engineering(named, target_size=11)
    feature_vec = apply_scaler(feature_vec, scaler_tuple)

    # Accuracy model adalah Dense (bukan GRU) — shape (1, 11)
    return feature_vec.reshape(1, 1, 11).astype(np.float32)    