from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes.assess import router as assess_router
from app.api.v1.routes.analyze import router as analyze_router
from app.api.v1.routes.feedback import router as feedback_router
from app.core.config import get_settings

settings = get_settings()

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan: preload models saat startup ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Dijalankan saat server startup dan shutdown.
    Preload semua model keras agar request pertama tidak lambat.
    """
    logger.info("=== Server startup: memuat model i-Speak ===")
    try:
        from app.services.ispeak.model_inference import model_loader

        model_loader.load_all()
        logger.info("=== Semua model i-Speak berhasil dimuat ===")
    except Exception as e:
        logger.error(f"Gagal memuat model i-Speak: {e}", exc_info=True)
        logger.warning(
            "Server tetap berjalan, tapi endpoint /analyze mungkin tidak berfungsi"
        )

    yield  # Server berjalan di sini

    # Cleanup saat shutdown (opsional)
    logger.info("Server shutdown.")


# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="i-Speak Diagnostic API",
    version="2.0.0",
    description=(
        "API untuk analisis kemampuan speaking bahasa Inggris berbasis CEFR. "
        "Mengintegrasikan model i-Speak (7 sub-construct) dengan pipeline RAG "
        "untuk menghasilkan feedback pembelajaran yang adaptif."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain frontend di produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(
    analyze_router,
    prefix="/api/v1",
    tags=["i-Speak Model"],
)

app.include_router(
    assess_router,
    prefix="/api/v1",
    tags=["RAG Assessment"],
)

app.include_router(
    feedback_router,
    prefix="/api/v1",
    tags=["Feedback"],
)


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "i-Speak Diagnostic API",
        "version": "2.0.0",
        "status": "running",
        "env": settings.app_env,
        "llm_backend": "ollama" if settings.use_local_llm else "dashscope",
        "embedding_model": settings.embedding_model,
        "endpoints": {
            "POST /api/v1/analyze": "Upload audio → prediksi 7 sub-construct CEFR",
            "GET  /api/v1/analyze/health": "Cek status model i-Speak",
            "POST /api/v1/assess": "ISpeakOutput JSON → feedback RAG",
            "GET  /api/v1/assess/mock": "Mock ISpeakOutput untuk testing frontend",
            "GET  /api/v1/feedback": "Endpoint feedback (sesuai isi feedback.py)",
            "GET  /api/v1/feedback/history": "Riwayat semua sesi feedback",
            "GET  /api/v1/feedback/{session_id}": "Detail satu sesi feedback",
            "POST /api/v1/feedback/{session_id}/rating": "Simpan rating pengguna",
            "GET  /api/v1/feedback/stats/summary": "Statistik agregat feedback",
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "env": settings.app_env,
        "llm_backend": "ollama" if settings.use_local_llm else "dashscope",
        "embedding_model": settings.embedding_model,
    }
