"""
app/core/security.py

Middleware keamanan ringan tanpa dependensi tambahan:
1. API key auth (header X-API-Key) — opsional, aktif via settings
2. Rate limiting sliding-window in-memory per client (IP atau API key)

Keduanya bisa dimatikan via settings agar dev tetap nyaman:
- API_KEY_AUTH_ENABLED=false (default)
- RATE_LIMIT_ENABLED=false (default)
"""

import logging
import time
from collections import defaultdict, deque
from typing import Deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ─── Endpoint yang bebas auth (public health / docs) ───────────────────────
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}


def _client_id(request: Request) -> str:
    """Identitas client: API key kalau ada, fallback ke IP."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"key:{api_key}"
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else request.client.host if request.client else "unknown"
    return f"ip:{ip}"


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Cek header X-API-Key terhadap settings.api_keys."""

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if not settings.api_key_auth_enabled:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key or api_key not in settings.api_keys:
            logger.warning(f"Unauthorized request ke {request.url.path} dari {request.client.host if request.client else '?'}")
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "API key tidak valid. Kirim header X-API-Key."},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window sederhana: N request per menit per client."""

    _hits: dict = defaultdict(deque)  # type: ignore[var-annotated]
    _lock_placeholder = True  # GIL cukup aman untuk kasus ini

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if not settings.rate_limit_enabled:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
            return await call_next(request)

        cid = _client_id(request)
        now = time.monotonic()
        window = 60.0
        limit = settings.rate_limit_per_minute

        queue: Deque[float] = self._hits[cid]
        # Buang entri di luar window
        while queue and now - queue[0] > window:
            queue.popleft()

        if len(queue) >= limit:
            logger.warning(f"Rate limit exceeded untuk {cid}")
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": f"Terlalu banyak request. Maksimal {limit} request/menit.",
                },
                headers={"Retry-After": str(int(window - (now - queue[0])))},
            )

        queue.append(now)
        return await call_next(request)
