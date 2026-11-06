"""
app/core/security.py

Authentication and rate-limiting primitives.

- ``require_api_key`` — FastAPI dependency that authenticates the ``X-API-Key``
  header (constant-time compare) and returns a ``Requester`` identity.
- ``APIKeyAuthMiddleware`` — blanket gate for endpoints that don't opt into the
  dependency directly (defence in depth).
- ``RateLimitMiddleware`` — per-client sliding-window limiter. In-memory only:
  each process keeps its own counters. For multi-worker deployments, put a
  shared store (e.g. Redis) behind ``get_rate_limit_store``.

Both features are disabled by default (``api_key_auth_enabled=false`` /
``rate_limit_enabled=false``) so local development and the existing test suite
behave exactly as before; enable them via environment in production.
"""

import hashlib
import logging
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Endpoints that are always public, regardless of auth setting. Kept explicit
# and narrow: health/readiness and the documentation UI. Documentation can be
# turned off separately via settings.docs_enabled.
PUBLIC_PATHS = frozenset({"/", "/health", "/favicon.ico"})


# ─── Identity ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Requester:
    """Authenticated caller identity derived from a valid API key."""

    client_id: str
    is_admin: bool = False
    key_hash: str = ""


def _hash_key(api_key: str) -> str:
    """One-way hash of an API key. Never store or log the raw key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _authenticate(api_key: Optional[str]) -> Requester:
    settings = get_settings()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Send the X-API-Key header.",
        )

    # Constant-time comparison against every configured key.
    for candidate in settings.admin_api_keys:
        if secrets.compare_digest(api_key, candidate):
            return Requester(
                client_id=_hash_key(candidate),
                is_admin=True,
                key_hash=_hash_key(candidate),
            )
    for candidate in settings.api_keys:
        if secrets.compare_digest(api_key, candidate):
            return Requester(
                client_id=_hash_key(candidate),
                is_admin=False,
                key_hash=_hash_key(candidate),
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key.",
    )


def require_api_key(request: Request) -> Requester:
    """FastAPI dependency: authenticate X-API-Key and return the caller.

    When auth is disabled (development/testing), returns an anonymous admin
    identity so existing behaviour is preserved.
    """
    settings = get_settings()
    if not settings.api_key_auth_enabled:
        return Requester(client_id="anonymous", is_admin=True, key_hash="anonymous")

    return _authenticate(request.headers.get("X-API-Key"))


# ─── API key middleware (defence in depth) ──────────────────────────────────


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid X-API-Key before they reach routing."""

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if not settings.api_key_auth_enabled:
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS or path == "/docs" or path == "/redoc" or path == "/openapi.json":
            if settings.docs_enabled or path in PUBLIC_PATHS:
                return await call_next(request)

        try:
            _authenticate(request.headers.get("X-API-Key"))
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"success": False, "error": exc.detail},
            )

        return await call_next(request)


# ─── Rate limiting ──────────────────────────────────────────────────────────


@dataclass
class _Window:
    hits: Deque[float] = field(default_factory=deque)


def _client_id(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    Uses the API key hash when present (so a client is limited by key, not IP),
    otherwise the socket peer. X-Forwarded-For is only honoured when the
    deployment is explicitly configured behind a trusted proxy.
    """
    settings = get_settings()

    api_key = request.headers.get("X-API-Key")
    if api_key and settings.api_key_auth_enabled:
        return f"key:{_hash_key(api_key)}"

    peer = request.client.host if request.client else "unknown"
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            peer = forwarded.split(",")[0].strip()

    return f"ip:{peer}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter, per client.

    In-memory only: the store lives on this middleware instance (per process).
    With multiple Uvicorn workers or replicas, each process has its own counters,
    so the effective limit scales with the worker count. For a hard multi-worker
    limit, pass a shared store (e.g. Redis-backed) via ``store=``.
    """

    def __init__(self, app, store: Optional[Dict[str, _Window]] = None):
        super().__init__(app)
        self._store: Dict[str, _Window] = (
            store if store is not None else defaultdict(_Window)
        )

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if not settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        if path.endswith("/analyze"):
            limit = settings.analyze_rate_limit_per_minute
        elif path.endswith("/assess"):
            limit = settings.assess_rate_limit_per_minute
        else:
            limit = settings.rate_limit_per_minute

        if limit <= 0:
            return await call_next(request)

        cid = _client_id(request)
        now = time.monotonic()
        window = 60.0
        queue = self._store[cid].hits

        while queue and now - queue[0] > window:
            queue.popleft()

        if len(queue) >= limit:
            retry_after = int(window - (now - queue[0])) if queue else int(window)
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Rate limit exceeded. Retry later.",
                },
                headers={"Retry-After": str(max(1, retry_after))},
            )

        queue.append(now)
        return await call_next(request)
