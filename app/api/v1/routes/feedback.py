"""
app/api/v1/routes/feedback.py

Feedback session storage and retrieval.

Ownership: every session is tagged with the authenticated caller (``owner_id``,
a hash of the API key). Regular clients can only see/modify their own sessions;
admin clients can list and manage all sessions. When API-key auth is disabled
(development/testing), ownership is ``"anonymous"`` and all sessions remain
visible, preserving the previous behaviour.

Storage is in-memory by design (see AGENTS.md) and does not survive restarts.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import Requester, require_api_key
from app.models.schemas import FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Local schemas ───────────────────────────────────────────────────────────
class FeedbackSession(BaseModel):
    session_id: str
    created_at: str
    overall_cefr: str
    gap_skills: list[str]
    owner_id: str = "anonymous"
    feedback: FeedbackResponse
    user_rating: Optional[int] = None
    user_notes: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class FeedbackHistoryItem(BaseModel):
    session_id: str
    created_at: str
    overall_cefr: str
    gap_skills: list[str]
    user_rating: Optional[int] = None


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating 1–5")
    notes: Optional[str] = Field(None, description="Catatan opsional dari pengguna")


# ─── In-memory store ─────────────────────────────────────────────────────────
_feedback_store: dict[str, FeedbackSession] = {}


def _save_session(
    feedback_response: FeedbackResponse,
    overall_cefr: str,
    gap_skills: list[str],
    owner_id: str = "anonymous",
) -> str:
    """Store a feedback response and return its session id."""
    session_id = str(uuid.uuid4())
    session = FeedbackSession(
        session_id=session_id,
        created_at=datetime.now().isoformat(),
        overall_cefr=overall_cefr,
        gap_skills=gap_skills,
        owner_id=owner_id,
        feedback=feedback_response,
    )
    _feedback_store[session_id] = session
    logger.info("Sesi feedback disimpan: %s (CEFR: %s)", session_id, overall_cefr)
    return session_id


def _owned_sessions(requester: Requester) -> list[FeedbackSession]:
    """All sessions the requester is allowed to see."""
    if requester.is_admin:
        return list(_feedback_store.values())
    return [s for s in _feedback_store.values() if s.owner_id == requester.client_id]


def _get_owned(session_id: str, requester: Requester) -> FeedbackSession:
    session = _feedback_store.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sesi feedback tidak ditemukan.",
        )
    if not requester.is_admin and session.owner_id != requester.client_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sesi feedback tidak ditemukan.",
        )
    return session


# ─── Endpoints ───────────────────────────────────────────────────────────────
@router.get(
    "/feedback/history",
    response_model=list[FeedbackHistoryItem],
    summary="Ambil riwayat sesi feedback milik pemanggil",
)
async def get_feedback_history(requester: Requester = Depends(require_api_key)):
    sessions = sorted(_owned_sessions(requester), key=lambda s: s.created_at, reverse=True)
    return [
        FeedbackHistoryItem(
            session_id=s.session_id,
            created_at=s.created_at,
            overall_cefr=s.overall_cefr,
            gap_skills=s.gap_skills,
            user_rating=s.user_rating,
        )
        for s in sessions
    ]


@router.get(
    "/feedback/{session_id}",
    response_model=FeedbackSession,
    summary="Ambil detail satu sesi feedback",
)
async def get_feedback_session(
    session_id: str, requester: Requester = Depends(require_api_key)
):
    return _get_owned(session_id, requester)


@router.post(
    "/feedback/{session_id}/rating",
    summary="Simpan rating dari pengguna",
)
async def rate_feedback(
    session_id: str,
    body: RatingRequest,
    requester: Requester = Depends(require_api_key),
):
    session = _get_owned(session_id, requester)
    session.user_rating = body.rating
    session.user_notes = body.notes
    _feedback_store[session_id] = session
    logger.info("Rating disimpan untuk sesi %s: %s/5", session_id, body.rating)
    return {
        "session_id": session_id,
        "rating": body.rating,
        "notes": body.notes,
        "message": "Rating berhasil disimpan.",
    }


@router.delete(
    "/feedback/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Hapus satu sesi feedback",
)
async def delete_feedback_session(
    session_id: str, requester: Requester = Depends(require_api_key)
):
    session = _get_owned(session_id, requester)
    del _feedback_store[session_id]
    logger.info("Sesi feedback dihapus: %s", session_id)
    return {"session_id": session_id, "message": "Sesi berhasil dihapus."}


@router.get(
    "/feedback/stats/summary",
    summary="Statistik ringkas sesi feedback",
)
async def get_feedback_stats(requester: Requester = Depends(require_api_key)):
    sessions = _owned_sessions(requester)
    if not sessions:
        return {
            "total_sessions": 0,
            "cefr_distribution": {},
            "top_gap_skills": [],
            "avg_rating": None,
            "rated_sessions": 0,
        }

    cefr_dist: dict[str, int] = {}
    for s in sessions:
        cefr_dist[s.overall_cefr] = cefr_dist.get(s.overall_cefr, 0) + 1

    gap_counter: dict[str, int] = {}
    for s in sessions:
        for gap in s.gap_skills:
            gap_counter[gap] = gap_counter.get(gap, 0) + 1
    top_gaps = sorted(gap_counter.items(), key=lambda x: x[1], reverse=True)

    rated = [s for s in sessions if s.user_rating is not None]
    avg_rat = round(sum(s.user_rating for s in rated) / len(rated), 2) if rated else None

    return {
        "total_sessions": len(sessions),
        "cefr_distribution": cefr_dist,
        "top_gap_skills": [{"skill": k, "count": v} for k, v in top_gaps[:5]],
        "avg_rating": avg_rat,
        "rated_sessions": len(rated),
    }
