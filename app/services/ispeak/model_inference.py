"""
app/services/ispeak/model_inference.py — UPDATED

Perubahan dari versi sebelumnya:
- Semua 7 scaler .pkl dimuat (bukan hanya Accuracy)
- Setiap extract_*_features() sekarang menerima scaler_tuple
- Scaler diaplikasikan di dalam feature_extractor (PowerTransformer + MinMaxScaler)
"""

import os, pickle, warnings, logging
import numpy as np
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
CEFR_LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


class ISpeakModelLoader:
    _instance = None
    _models: dict = {}
    _scalers: dict = {}
    _loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_all(self):
        if self._loaded:
            return
        import tensorflow as tf

        model_files = {
            "Fluency": "Fluency_final.keras",
            "Pronunciation": "Pronunciation_final.keras",
            "Prosody": "Prosody_final.keras",
            "Coherence_and_Cohesion": "Coherence_and_Cohesion_final.keras",
            "Topic_Relevance": "Topic_Relevance_final.keras",
            "Complexity": "Complexity_final.keras",
            "Accuracy": "Accuracy_final.keras",
        }
        scaler_files = {
            "Fluency": "Fluency_final_scaler.pkl",
            "Pronunciation": "Pronunciation_final_scaler.pkl",
            "Prosody": "Prosody_final_scaler.pkl",
            "Coherence_and_Cohesion": "Coherence_and_Cohesion_final_scaler.pkl",
            "Topic_Relevance": "Topic_Relevance_final_scaler.pkl",
            "Complexity": "Complexity_final_scaler.pkl",
            "Accuracy": "Accuracy_final_scaler.pkl",
        }

        # ── Load model dulu (skip scaler sementara) ──
        for name, fname in model_files.items():
            path = MODELS_DIR / fname
            if not path.exists():
                raise FileNotFoundError(f"Model tidak ditemukan: {path}")
            try:
                self._models[name] = tf.keras.models.load_model(
                    str(path),
                    compile=False,  # ← tambah ini
                )
                logger.info(f"✓ Model {name} dimuat")
            except Exception as e:
                logger.error(f"✗ Model {name} gagal: {e}", exc_info=True)
                raise

        # ── Load scaler terpisah ──
        for name, fname in scaler_files.items():
            path = MODELS_DIR / fname
            if not path.exists():
                logger.warning(f"Scaler {name} tidak ditemukan, skip")
                continue
            try:
                with open(path, "rb") as f:
                    self._scalers[name] = pickle.load(f)
                logger.info(f"✓ Scaler {name} dimuat")
            except Exception as e:
                logger.error(f"✗ Scaler {name} gagal: {e}", exc_info=True)
                raise

        self._loaded = True
        logger.info("Semua model & scaler i-Speak berhasil dimuat")

    def get_model(self, name: str):
        if not self._loaded:
            self.load_all()
        return self._models[name]

    def get_scaler(self, name: str):
        if not self._loaded:
            self.load_all()
        return self._scalers[name]

    @property
    def is_loaded(self):
        return self._loaded


model_loader = ISpeakModelLoader()


def _predict_cefr(model, feature_vec: np.ndarray) -> dict:
    import tensorflow as tf
    # Convert ke tensor eksplisit agar tidak retrace
    tensor_input = tf.constant(feature_vec, dtype=tf.float32)
    pred = model.predict(tensor_input, verbose=0)
    probs = pred[0].tolist()
    idx = int(np.argmax(probs))
    return {
        "cefr_level": CEFR_LABELS[idx],
        "confidence": round(float(probs[idx]), 4),
        "distribution": {l: round(float(p), 4) for l, p in zip(CEFR_LABELS, probs)},
    }


async def run_ispeak_inference(
    audio_path: str,
    transcript: Optional[str] = None,
    prompt_text: str = "",
) -> dict:
    # ═══════════════════════════════════════════════════════════════════════════
    # MOCK MODE: Bypass model ML untuk development frontend
    # ═══════════════════════════════════════════════════════════════════════════
    from app.core.config import get_settings

    settings = get_settings()

    if settings.ispeak_mock_mode:
        logger.warning("🧪 MOCK MODE: Mengembalikan dummy prediction tanpa model")

        # Transkripsi tetap jalan kalau tidak dikirim manual
        if not transcript:
            from .feature_extractor import transcribe_audio

            transcript = transcribe_audio(audio_path)

        return {
            "task_type": "FreeSpeech",
            "key_features": {
                "duration": 60.0,
                "total_words": 120,
                "wpm": 120.0,
                "grammar_errors": 3,
                "ttr": 0.65,
                "idioms_found": 1,
                "cefr_distribution": {
                    "A1": 10,
                    "A2": 20,
                    "B1": 40,
                    "B2": 30,
                    "C1": 15,
                    "C2": 5,
                },
            },
            "subconstructs": [
                {"name": "Fluency", "cefr_level": "B1", "confidence": 0.82},
                {"name": "Pronunciation", "cefr_level": "B1", "confidence": 0.79},
                {"name": "Prosody", "cefr_level": "B2", "confidence": 0.75},
                {
                    "name": "Coherence_and_Cohesion",
                    "cefr_level": "B1",
                    "confidence": 0.80,
                },
                {"name": "Topic_Relevance", "cefr_level": "B2", "confidence": 0.77},
                {"name": "Complexity", "cefr_level": "B1", "confidence": 0.73},
                {"name": "Accuracy", "cefr_level": "B1", "confidence": 0.81},
            ],
            "transcript": transcript,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # KODE ASLI
    # ═══════════════════════════════════════════════════════════════════════════
    from .feature_extractor import (
        transcribe_audio,
        extract_fluency_features,
        extract_pronunciation_features,
        extract_prosody_features,
        extract_coherence_features,
        extract_topic_relevance_features,
        extract_complexity_features,
        extract_accuracy_features,
    )
    import librosa

    # 1. Transkripsi
    if not transcript:
        logger.info("Menjalankan transkripsi Whisper...")
        transcript = transcribe_audio(audio_path)
    logger.info(f"Transcript ({len(transcript.split())} words): {transcript[:60]}...")

    # 2. KeyFeatures
    y, sr = librosa.load(audio_path, sr=16000)
    duration = float(librosa.get_duration(y=y, sr=sr))
    words = transcript.strip().split()
    total_w = len(words)
    wpm = (total_w / duration * 60) if duration > 0 else 0.0
    alpha = [w.lower().strip(".,!?") for w in words if w.isalpha()]
    ttr = len(set(alpha)) / max(len(alpha), 1)

    # Grammar errors
    try:
        import language_tool_python

        tool = language_tool_python.LanguageTool("en-US")
        grammar_errors = len(tool.check(transcript))
        tool.close()
    except Exception:
        grammar_errors = 0

    IDIOM_LIST = {
        "on the other hand",
        "in the long run",
        "at the end of the day",
        "break a leg",
        "cost an arm and a leg",
        "hit the nail on the head",
    }
    idioms_found = sum(1 for idiom in IDIOM_LIST if idiom in transcript.lower())

    BASIC = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
    }

    def cefr_dist(ws):
        d = {"A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0, "C2": 0}
        for w in ws:
            if w in BASIC:
                d["A1"] += 1
            elif len(w) <= 5:
                d["A2"] += 1
            elif len(w) <= 7:
                d["B1"] += 1
            elif len(w) <= 9:
                d["B2"] += 1
            elif len(w) <= 11:
                d["C1"] += 1
            else:
                d["C2"] += 1
        return d

    key_features = {
        "duration": round(duration, 2),
        "total_words": total_w,
        "wpm": round(wpm, 2),
        "grammar_errors": grammar_errors,
        "ttr": round(ttr, 4),
        "idioms_found": idioms_found,
        "cefr_distribution": cefr_dist(alpha),
    }

    # 3. Prediksi 7 sub-construct
    results = {}

    logger.info("Prediksi Fluency...")
    results["Fluency"] = _predict_cefr(
        model_loader.get_model("Fluency"),
        extract_fluency_features(
            audio_path, transcript, model_loader.get_scaler("Fluency")
        ),
    )

    logger.info("Prediksi Pronunciation...")
    results["Pronunciation"] = _predict_cefr(
        model_loader.get_model("Pronunciation"),
        extract_pronunciation_features(
            audio_path, transcript, model_loader.get_scaler("Pronunciation")
        ),
    )

    logger.info("Prediksi Prosody...")
    results["Prosody"] = _predict_cefr(
        model_loader.get_model("Prosody"),
        extract_prosody_features(audio_path, model_loader.get_scaler("Prosody")),
    )

    logger.info("Prediksi Coherence & Cohesion...")
    results["Coherence_and_Cohesion"] = _predict_cefr(
        model_loader.get_model("Coherence_and_Cohesion"),
        extract_coherence_features(
            transcript, model_loader.get_scaler("Coherence_and_Cohesion")
        ),
    )

    logger.info("Prediksi Topic Relevance...")
    results["Topic_Relevance"] = _predict_cefr(
        model_loader.get_model("Topic_Relevance"),
        extract_topic_relevance_features(
            transcript, model_loader.get_scaler("Topic_Relevance"), prompt_text
        ),
    )

    logger.info("Prediksi Complexity...")
    results["Complexity"] = _predict_cefr(
        model_loader.get_model("Complexity"),
        extract_complexity_features(transcript, model_loader.get_scaler("Complexity")),
    )

    logger.info("Prediksi Accuracy...")
    results["Accuracy"] = _predict_cefr(
        model_loader.get_model("Accuracy"),
        extract_accuracy_features(
            audio_path, transcript, model_loader.get_scaler("Accuracy")
        ),
    )

    # 4. Build output
    ispeak_output = {
        "task_type": "FreeSpeech",
        "key_features": key_features,
        "subconstructs": [
            {"name": name, "cefr_level": r["cefr_level"], "confidence": r["confidence"]}
            for name, r in results.items()
        ],
        "transcript": transcript,
    }

    logger.info(f"ISpeakOutput: { {k: v['cefr_level'] for k, v in results.items()} }")
    return ispeak_output
