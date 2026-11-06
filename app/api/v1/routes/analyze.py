"""
app/api/v1/routes/analyze.py

Endpoint POST /api/v1/analyze
Menerima file audio → ekstrak fitur → jalankan 7 model i-Speak
→ kembalikan ISpeakOutput JSON yang siap diteruskan ke /api/v1/assess
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.security import Requester, require_api_key
from app.services.ispeak.model_inference import run_ispeak_inference

logger = logging.getLogger(__name__)
router = APIRouter()

settings = get_settings()

# ─── Upload staging ──────────────────────────────────────────────────────────
def _staging_dir() -> Path:
    """Upload staging directory, created on demand. UUID filenames are used, so
    client filenames never become paths on disk."""
    try:
        d = Path(settings.upload_temp_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        import tempfile

        return Path(tempfile.gettempdir()) / "ispeak_audio"


# MIME types accepted for audio uploads. "application/octet-stream" and empty are
# tolerated as "unknown" (some clients send them); everything else that is not
# audio is rejected. Extension and magic-byte checks are the real gate.
_ALLOWED_MIME = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3", "audio/mpga",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac",
    "audio/ogg", "audio/opus", "application/ogg",
    "application/octet-stream", "",
}
_KNOWN_NON_AUDIO_PREFIXES = ("text/", "image/", "video/", "application/json", "application/xml", "application/pdf")

# ─── Container sniffing ──────────────────────────────────────────────────────
def _sniff_audio_format(path: Path) -> str | None:
    """Return a coarse container type from magic bytes, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None

    if len(head) < 12:
        return None
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head[4:8] == b"ftyp":
        return "mp4"
    return None


# Allowed sniffed containers must be consistent with the allowed extensions.
_SNIFF_TO_EXT = {"wav": {".wav"}, "ogg": {".ogg"}, "flac": {".flac"},
                 "mp3": {".mp3"}, "mp4": {".m4a", ".mp4"}}


def _probe_duration(path: Path) -> float | None:
    """Cheap audio duration probe (header-only where possible)."""
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        pass
    try:
        import audioread

        with audioread.audio_open(str(path)) as f:
            return float(f.duration)
    except Exception:
        return None


# ─── Endpoint ────────────────────────────────────────────────────────────────
@router.post(
    "/analyze",
    summary="Analisis audio speaking dengan model i-Speak",
    description=(
        "Upload file audio WAV/MP3, sistem akan melakukan transkripsi, "
        "ekstraksi fitur, dan prediksi 7 sub-construct CEFR menggunakan "
        "model i-Speak. Output dapat langsung diteruskan ke endpoint /assess."
    ),
)
async def analyze_audio(
    audio: UploadFile = File(..., description="File audio WAV atau MP3"),
    prompt_text: str = Form(default="", description="Teks pertanyaan/prompt (opsional)"),
    transcript: str = Form(default="", description="Transkripsi manual (opsional)"),
    requester: Requester = Depends(require_api_key),
):
    # ── 1. Extension validation (never trust client filename as path) ──────
    client_name = audio.filename or ""
    suffix = Path(client_name).suffix.lower()
    if suffix not in settings.allowed_audio_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Format file tidak didukung: {suffix or '(tanpa ekstensi)'}. "
                   f"Gunakan salah satu dari {', '.join(settings.allowed_audio_extensions)}.",
        )

    # ── 2. MIME validation (reject obviously non-audio types) ──────────────
    content_type = (audio.content_type or "").lower().split(";")[0].strip()
    if content_type.startswith(_KNOWN_NON_AUDIO_PREFIXES):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Format file tidak didukung.",
        )
    if content_type not in _ALLOWED_MIME and not content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Format file tidak didukung.",
        )

    # ── 3. Transcript length cap ────────────────────────────────────────────
    if len(transcript) > settings.max_transcript_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Transkrip terlalu panjang (maksimum {settings.max_transcript_chars} karakter).",
        )

    # ── 4. Stream to disk with a size cap enforced DURING the write ────────
    staging = _staging_dir()
    temp_path = staging / f"{uuid.uuid4()}{suffix}"
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    total = 0

    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"Ukuran file terlalu besar. "
                            f"Maksimal {settings.max_upload_size_mb} MB."
                        ),
                    )
                f.write(chunk)

        if total < 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File audio terlalu kecil atau kosong.",
            )

        # ── 5. Magic-byte validation ───────────────────────────────────────
        sniffed = _sniff_audio_format(temp_path)
        if sniffed is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File bukan audio yang valid.",
            )
        if suffix not in _SNIFF_TO_EXT.get(sniffed, set()):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Konten file tidak cocok dengan ekstensi.",
            )

        # ── 6. Duration limit ───────────────────────────────────────────────
        duration = _probe_duration(temp_path)
        if duration is not None and duration > settings.audio_max_duration_sec:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Durasi audio terlalu panjang ({duration:.0f} detik). "
                    f"Maksimal {settings.audio_max_duration_sec} detik."
                ),
            )

        logger.info(
            "Audio diterima | size=%.2f MB | duration=%s | format=%s",
            total / (1024 * 1024),
            round(duration, 2) if duration is not None else "unknown",
            sniffed,
        )

        # ── 7. Inference ────────────────────────────────────────────────────
        ispeak_output = await run_ispeak_inference(
            audio_path=str(temp_path),
            transcript=transcript.strip() or None,
            prompt_text=prompt_text.strip(),
        )

        return JSONResponse(content=ispeak_output)

    except HTTPException:
        raise
    except FileNotFoundError:
        logger.error("Model file tidak ditemukan.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model i-Speak belum tersedia.",
        )
    except RuntimeError:
        logger.error("Inference error.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal menjalankan analisis.",
        )
    except Exception:
        logger.exception("Unexpected error di /analyze")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan internal.",
        )
    finally:
        # Always remove the temporary file, even on error.
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            logger.warning("Gagal menghapus file sementara: %s", temp_path)


@router.get("/analyze/health", summary="Cek status model i-Speak")
async def analyze_health():
    """Cek apakah semua model i-Speak sudah dimuat dan siap digunakan."""
    from app.services.ispeak.model_inference import model_loader

    if not model_loader.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": "Model i-Speak belum dimuat."},
        )

    # Do not disclose the internal model names; report readiness only.
    return {
        "status": "ready",
        "models_loaded": len(model_loader._models),
        "message": "Semua model i-Speak siap digunakan.",
    }
