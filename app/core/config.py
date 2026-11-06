from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import List, Optional


class Settings(BaseSettings):
    # ─── Existing: Supabase ─────────────────────────────────────────────────
    supabase_url: str
    supabase_key: str

    # ─── Existing: LLM ──────────────────────────────────────────────────────
    use_local_llm: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:72b"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen2.5-72b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")

    # ─── Existing: Embedding & RAG ──────────────────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.75
    max_context_tokens: int = 3000
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ─── Existing: Langfuse ─────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ─── Existing: App ──────────────────────────────────────────────────────
    app_env: str = "development"
    app_port: int = 8000

    # ═══ TAMBAHAN: i-Speak Model ════════════════════════════════════════════
    ispeak_model_dir: str = "./models/ispeak"           # Folder 7 model .keras/.h5
    ispeak_device: str = "cpu"                          # "cpu" atau "cuda"
    ispeak_mock_mode: bool = False                      # True = bypass model load (dev/testing)
    ispeak_confidence_threshold: float = 0.6            # Threshold prediksi valid

    # ═══ TAMBAHAN: Audio Upload ═════════════════════════════════════════════
    max_upload_size_mb: int = 50
    allowed_audio_extensions: List[str] = [".wav", ".mp3", ".m4a", ".ogg"]
    upload_temp_dir: str = "./tmp/uploads"              # Temp file sebelum diproses
    audio_sample_rate: int = 16000                      # Sample rate untuk feature extraction
    audio_max_duration_sec: int = 300                   # Maksimal 5 menit

    # ═══ CORS ═══════════════════════════════════════════════════════════════
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"],
        description="Origin yang diizinkan. Set CORS_ORIGINS=* untuk semua (tidak disarankan di produksi).",
    )

    # ═══ Security: API key & rate limiting ═══════════════════════════════════
    api_key_auth_enabled: bool = False
    api_keys: List[str] = Field(
        default_factory=list,
        description="Daftar API key yang valid (header X-API-Key). Kosong + enabled=False = tanpa auth.",
    )
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 30     # Request per menit per client (analyze/assess)
    rate_limit_burst: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()