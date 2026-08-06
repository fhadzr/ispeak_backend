"""
app/api/v1/routes/analyze.py

Endpoint POST /api/v1/analyze
Menerima file audio → ekstrak fitur → jalankan 7 model i-Speak
→ kembalikan ISpeakOutput JSON yang siap diteruskan ke /api/v1/assess
"""

import os
import uuid
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from app.services.ispeak.model_inference import run_ispeak_inference

logger = logging.getLogger(__name__)
router = APIRouter()

# Folder sementara untuk file audio yang diupload
TEMP_DIR = Path(tempfile.gettempdir()) / "ispeak_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm",
    "application/octet-stream",  # fallback
}
MAX_FILE_SIZE_MB = 50


@router.post(
    "/analyze",
    summary="Analisis audio speaking dengan model i-Speak",
    description=(
        "Upload file audio WAV/MP3, sistem akan melakukan transkripsi, "
        "ekstraksi fitur, dan prediksi 7 sub-construct CEFR menggunakan "
        "model i-Speak. Output dapat langsung diteruskan ke endpoint /assess."
    ),
    response_description="ISpeakOutput JSON dengan 7 SubConstructScore dan KeyFeatures",
)
async def analyze_audio(
    audio: UploadFile = File(..., description="File audio WAV atau MP3"),
    prompt_text: str = Form(
        default="",
        description="Teks pertanyaan/prompt (opsional, digunakan untuk Topic Relevance)",
    ),
    transcript: str = Form(
        default="",
        description="Transkripsi manual (opsional, jika kosong akan digenerate via Whisper)",
    ),
):
    """
    Pipeline lengkap analisis speaking:
    1. Validasi file audio
    2. Simpan sementara ke disk
    3. Jalankan transkripsi Whisper (jika tidak ada transcript)
    4. Ekstrak fitur untuk 7 sub-construct
    5. Prediksi CEFR per sub-construct
    6. Kembalikan ISpeakOutput

    Response format:
    ```json
    {
      "task_type": "FreeSpeech",
      "key_features": { "duration": 41.98, "wpm": 147.23, ... },
      "subconstructs": [
        { "name": "Fluency", "cefr_level": "B2", "confidence": 0.963 },
        ...
      ],
      "transcript": "..."
    }
    ```
    """

    # ── Validasi content type ─────────────────────────────────────────────
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Format file tidak didukung: {audio.content_type}. "
                f"Gunakan WAV atau MP3."
            ),
        )

    # ── Baca & validasi ukuran file ────────────────────────────────────────
    audio_bytes = await audio.read()
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Ukuran file terlalu besar: {size_mb:.1f} MB. Maksimal {MAX_FILE_SIZE_MB} MB.",
        )

    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File audio terlalu kecil atau kosong.",
        )

    # ── Simpan ke file sementara ───────────────────────────────────────────
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    temp_path = TEMP_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Audio disimpan sementara di {temp_path} ({size_mb:.2f} MB)")

        # ── Jalankan inference ─────────────────────────────────────────────
        ispeak_output = await run_ispeak_inference(
            audio_path=str(temp_path),
            transcript=transcript.strip() or None,
            prompt_text=prompt_text.strip(),
        )

        return JSONResponse(content=ispeak_output)

    except FileNotFoundError as e:
        logger.error(f"Model file tidak ditemukan: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model i-Speak belum tersedia: {str(e)}",
        )
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal menjalankan analisis: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error di /analyze: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan internal. Cek log server.",
        )
    finally:
        # Selalu hapus file sementara
        if temp_path.exists():
            temp_path.unlink()
            logger.debug(f"File sementara dihapus: {temp_path}")


@router.get("/analyze/health", summary="Cek status model i-Speak")
async def analyze_health():
    """Cek apakah semua model i-Speak sudah dimuat dan siap digunakan."""
    from app.services.ispeak.model_inference import model_loader

    if not model_loader.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Model i-Speak belum dimuat. Restart server atau tunggu startup selesai.",
            },
        )

    return {
        "status": "ready",
        "models_loaded": list(model_loader._models.keys()),
        "scaler_loaded": len(model_loader._scalers) == 7,
        "message": "Semua model i-Speak siap digunakan.",
    }