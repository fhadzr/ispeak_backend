"""Application settings.

Loaded from environment variables and an optional ``.env`` file. All secrets are
read from the environment (never hardcoded) and defaults are safe for local
development. Production validation happens explicitly via
:meth:`Settings.validate_for_startup`.
"""

import json
from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Environment ───────────────────────────────────────────────────────
    app_env: Literal["development", "testing", "production"] = "development"
    app_port: int = 8000

    # ─── Supabase ──────────────────────────────────────────────────────────
    # Empty by default so the app can be imported without a live .env (tests,
    # docs). validate_for_startup() enforces presence in production.
    supabase_url: str = ""
    supabase_key: str = ""

    # ─── LLM / Groq ────────────────────────────────────────────────────────
    # The pipeline currently talks to Groq (see app/services/llm_engine.py).
    groq_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500
    # Maximum characters of learner transcript accepted into the LLM prompts.
    max_transcript_chars: int = 20_000

    # ─── Embedding & RAG ───────────────────────────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.75
    max_context_tokens: int = 3000
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ─── i-Speak Model ─────────────────────────────────────────────────────
    ispeak_model_dir: str = "./models/ispeak"
    ispeak_device: str = "cpu"
    ispeak_mock_mode: bool = False
    ispeak_confidence_threshold: float = 0.6

    # ─── Audio Upload ──────────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    allowed_audio_extensions: List[str] = [".wav", ".mp3", ".m4a", ".ogg"]
    upload_temp_dir: str = "./tmp/uploads"
    audio_sample_rate: int = 16000
    audio_max_duration_sec: int = 300

    # ─── CORS ──────────────────────────────────────────────────────────────
    # Restrictive default. Accepts a JSON array or a comma-separated list via
    # the environment (see validator below).
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"],
        description="Allowed origins. Set CORS_ORIGINS='https://a.com,https://b.com' "
                    "or CORS_ORIGINS='[\"https://a.com\"]'.",
    )

    # ─── Security: API key auth & rate limiting ────────────────────────────
    api_key_auth_enabled: bool = False
    # Regular client keys.
    api_keys: List[str] = Field(default_factory=list)
    # Admin keys: pass authorization checks that regular keys fail
    # (e.g. listing/deleting other clients' feedback).
    admin_api_keys: List[str] = Field(default_factory=list)

    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 60
    # Stricter limits for the expensive ML/LLM endpoints.
    analyze_rate_limit_per_minute: int = 10
    assess_rate_limit_per_minute: int = 10

    # When behind a trusted reverse proxy that overwrites X-Forwarded-For,
    # set this True so the rate limiter uses the forwarded client IP.
    trust_proxy_headers: bool = False

    # Documentation endpoints (Swagger UI / ReDoc / OpenAPI JSON).
    docs_enabled: bool = True

    # ─── CSV / comma-list parsing for list fields ──────────────────────────
    @field_validator(
        "cors_origins",
        "api_keys",
        "admin_api_keys",
        "allowed_audio_extensions",
        mode="before",
    )
    @classmethod
    def _parse_csv_or_json(cls, v):
        """Accept either a JSON array string or a plain comma-separated list."""
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in s.split(",") if item.strip()]
        return v

    def validate_for_startup(self) -> None:
        """Raise if the configuration is unsafe for the declared environment.

        Called from the FastAPI lifespan hook so misconfiguration fails loudly
        at boot rather than at first request.
        """
        if self.app_env == "production":
            missing = []
            if not self.supabase_url:
                missing.append("SUPABASE_URL")
            if not self.supabase_key:
                missing.append("SUPABASE_KEY")
            if not self.groq_api_key:
                missing.append("GROQ_API_KEY")
            if self.ispeak_mock_mode:
                missing.append("ISPEAK_MOCK_MODE must be false in production")
            if self.api_key_auth_enabled and not (self.api_keys or self.admin_api_keys):
                missing.append(
                    "API_KEY_AUTH_ENABLED=true requires API_KEYS or ADMIN_API_KEYS"
                )
            if missing:
                raise RuntimeError(
                    "Production configuration incomplete or unsafe: "
                    + "; ".join(missing)
                )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
