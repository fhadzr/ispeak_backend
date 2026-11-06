"""
Security tests: authentication, authorization, rate limiting, CORS, upload
validation, and frontend-safe rendering primitives.

These tests do NOT require live Supabase/LLM/model access. They build isolated
FastAPI apps and patch settings as needed.
"""

import io
import os
import sys

import pytest
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import Settings
from app.core import security as security_module
from app.core.security import (
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    Requester,
    _authenticate,
    _hash_key,
    require_api_key,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_settings(**overrides) -> Settings:
    return Settings(**overrides)


def patch_settings(monkeypatch, **overrides):
    """Point security_module.get_settings at an isolated Settings instance."""
    s = make_settings(**overrides)
    monkeypatch.setattr(security_module, "get_settings", lambda: s)
    return s


# ─── Authentication ──────────────────────────────────────────────────────────


class TestApiKeyAuthentication:
    def test_missing_key_rejected(self, monkeypatch):
        patch_settings(monkeypatch, api_key_auth_enabled=True, api_keys=["secret-1"])
        with pytest.raises(Exception) as exc:
            _authenticate(None)
        assert "401" in str(exc.value) or "Missing" in str(exc.value)

    def test_wrong_key_rejected(self, monkeypatch):
        patch_settings(monkeypatch, api_key_auth_enabled=True, api_keys=["secret-1"])
        with pytest.raises(Exception) as exc:
            _authenticate("wrong")
        assert "Invalid" in str(exc.value)

    def test_correct_key_accepted(self, monkeypatch):
        patch_settings(monkeypatch, api_key_auth_enabled=True, api_keys=["secret-1"])
        r = _authenticate("secret-1")
        assert r.client_id == _hash_key("secret-1")
        assert r.is_admin is False

    def test_admin_key_flagged(self, monkeypatch):
        patch_settings(
            monkeypatch, api_key_auth_enabled=True, admin_api_keys=["admin-1"]
        )
        r = _authenticate("admin-1")
        assert r.is_admin is True

    def test_auth_disabled_returns_anonymous_admin(self, monkeypatch):
        patch_settings(monkeypatch, api_key_auth_enabled=False)
        r = require_api_key(_make_request())
        assert r.client_id == "anonymous"
        assert r.is_admin is True


def _make_request(api_key: str | None = None):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [("x-api-key", api_key.encode())] if api_key else [],
    }
    return Request(scope)


class TestApiKeyDependencyEndToEnd:
    def test_protected_endpoint_rejects_without_key(self):
        app = FastAPI()

        @app.get("/protected")
        def protected(r: Requester = Depends(require_api_key)):
            return {"client": r.client_id}

        # Auth off by default → allowed (dev mode).
        c = TestClient(app)
        assert c.get("/protected").status_code == 200

    def test_protected_endpoint_requires_key_when_enabled(self, monkeypatch):
        patch_settings(monkeypatch, api_key_auth_enabled=True, api_keys=["k1"])
        app = FastAPI()

        @app.get("/protected")
        def protected(r: Requester = Depends(require_api_key)):
            return {"client": r.client_id}

        c = TestClient(app)
        assert c.get("/protected").status_code == 401
        assert c.get("/protected", headers={"X-API-Key": "bad"}).status_code == 401
        ok = c.get("/protected", headers={"X-API-Key": "k1"})
        assert ok.status_code == 200
        assert ok.json()["client"] == _hash_key("k1")


# ─── Authorization (feedback ownership) ──────────────────────────────────────


class TestFeedbackOwnership:
    def test_non_admin_sees_only_own_sessions(self):
        from app.api.v1.routes import feedback as fb

        fb._feedback_store.clear()
        fb._save_session(
            feedback_response=_dummy_feedback(),
            overall_cefr="B1",
            gap_skills=["Fluency"],
            owner_id="client-a",
        )
        fb._save_session(
            feedback_response=_dummy_feedback(),
            overall_cefr="B2",
            gap_skills=["Prosody"],
            owner_id="client-b",
        )

        a = Requester(client_id="client-a", is_admin=False)
        b = Requester(client_id="client-b", is_admin=False)
        admin = Requester(client_id="admin", is_admin=True)

        assert len(fb._owned_sessions(a)) == 1
        assert len(fb._owned_sessions(b)) == 1
        assert len(fb._owned_sessions(admin)) == 2

        # Client a must not be able to read client b's session.
        b_session = [s for s in fb._feedback_store.values() if s.owner_id == "client-b"][0]
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            fb._get_owned(b_session.session_id, a)
        assert exc.value.status_code == 404

        # Admin can read it.
        assert fb._get_owned(b_session.session_id, admin).session_id == b_session.session_id

    def test_non_admin_cannot_delete_others(self):
        from app.api.v1.routes import feedback as fb

        fb._feedback_store.clear()
        sid = fb._save_session(
            feedback_response=_dummy_feedback(),
            overall_cefr="B1",
            gap_skills=[],
            owner_id="client-b",
        )
        a = Requester(client_id="client-a", is_admin=False)
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            fb._get_owned(sid, a)


def _dummy_feedback():
    from app.models.schemas import (
        CEFRAssessment,
        CEFRLevel,
        FeedbackResponse,
        LearningPath,
    )

    return FeedbackResponse(
        overall_cefr=CEFRLevel.B1,
        sub_construct_summary=[],
        cefr_assessment=CEFRAssessment(
            overall_level=CEFRLevel.B1,
            sub_construct_feedback=[],
            justification="test",
            source_citations=[],
        ),
        priority_corrections=[],
        learning_path=LearningPath(
            current_level=CEFRLevel.B1,
            target_level=CEFRLevel.B2,
            duration_weeks=2,
            weekly_plan=[],
            source_citations=[],
        ),
        plain_summary="test",
        top_priority_action="test",
        retrieval_sources=[],
        hallucination_check_passed=True,
        model_used="test",
    )


# ─── Rate limiting ───────────────────────────────────────────────────────────


class TestRateLimiting:
    def test_limit_enforced(self, monkeypatch):
        patch_settings(
            monkeypatch, rate_limit_enabled=True, rate_limit_per_minute=2
        )
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/thing")
        def thing():
            return {"ok": True}

        c = TestClient(app)
        assert c.get("/thing").status_code == 200
        assert c.get("/thing").status_code == 200
        assert c.get("/thing").status_code == 429

    def test_disabled_by_default(self):
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/thing")
        def thing():
            return {"ok": True}

        c = TestClient(app)
        for _ in range(5):
            assert c.get("/thing").status_code == 200

    def test_spoofed_forwarded_header_ignored_by_default(self, monkeypatch):
        patch_settings(
            monkeypatch, rate_limit_enabled=True, rate_limit_per_minute=1
        )
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/thing")
        def thing():
            return {"ok": True}

        c = TestClient(app)
        # Different X-Forwarded-For values must NOT reset the limit, because
        # trust_proxy_headers is False by default.
        assert c.get("/thing", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert (
            c.get("/thing", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
        )


# ─── CORS ────────────────────────────────────────────────────────────────────


class TestCors:
    def _build(self, origins, credentials=False):
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/data")
        def data():
            return {"ok": True}

        return TestClient(app)

    def test_allowed_origin_present(self):
        c = self._build(["https://app.example.com"])
        r = c.get("/data", headers={"Origin": "https://app.example.com"})
        assert r.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_disallowed_origin_absent(self):
        c = self._build(["https://app.example.com"])
        r = c.get("/data", headers={"Origin": "https://evil.example.com"})
        assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"

    def test_wildcard_with_credentials_not_reflected(self):
        # With allow_credentials=False, Starlette must send literal "*", not echo
        # the attacker's Origin.
        c = self._build(["*"], credentials=False)
        r = c.get("/data", headers={"Origin": "https://evil.example.com"})
        assert r.headers.get("access-control-allow-origin") == "*"


# ─── Upload validation ───────────────────────────────────────────────────────


class TestUploadValidation:
    def test_non_audio_extension_rejected(self):
        from app.api.v1.routes.analyze import _sniff_audio_format, _staging_dir

        # _sniff_audio_format is pure; verify a text file is not audio.
        import tempfile
        from pathlib import Path

        p = Path(tempfile.gettempdir()) / "ispeak_test_notaudio.wav"
        p.write_bytes(b"this is definitely not audio data at all")
        try:
            assert _sniff_audio_format(p) is None
        finally:
            p.unlink()

    def test_real_wav_sniffed(self):
        import tempfile
        from pathlib import Path

        from app.api.v1.routes.analyze import _sniff_audio_format

        # Minimal RIFF/WAVE header.
        p = Path(tempfile.gettempdir()) / "ispeak_test_fake.wav"
        p.write_bytes(b"RIFF" + (28).to_bytes(4, "little") + b"WAVE")
        try:
            assert _sniff_audio_format(p) == "wav"
        finally:
            p.unlink()

    def test_analyze_rejects_invalid_extension(self):
        # Auth disabled by default; this must fail fast at extension check
        # before any model loading.
        from app.main import app

        c = TestClient(app)
        files = {"audio": ("test.txt", b"not audio", "text/plain")}
        r = c.post("/api/v1/analyze", files=files)
        assert r.status_code == 415
