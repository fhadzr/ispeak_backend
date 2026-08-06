"""
app/api/v1/routes/feedback.py — FIXED

Perubahan:
- FeedbackResponse lokal DIHAPUS — sekarang pakai dari app.models.schemas
- FeedbackSession.feedback sekarang pakai FeedbackResponse dari schemas
- _save_session menerima FeedbackResponse object langsung (bukan dict)
- Semua endpoint tetap sama
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.models.schemas import FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMAS LOKAL (hanya yang tidak ada di schemas.py)
# ═══════════════════════════════════════════════════════════════════════════

class FeedbackSession(BaseModel):
    """Satu sesi feedback lengkap yang disimpan."""
    session_id:   str
    created_at:   str
    overall_cefr: str
    gap_skills:   list[str]
    feedback:     FeedbackResponse
    user_rating:  Optional[int] = None
    user_notes:   Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class FeedbackHistoryItem(BaseModel):
    """Item ringkas untuk daftar riwayat."""
    session_id:   str
    created_at:   str
    overall_cefr: str
    gap_skills:   list[str]
    user_rating:  Optional[int] = None


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating 1–5")
    notes:  Optional[str] = Field(None, description="Catatan opsional dari pengguna")


# ═══════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORE
# ═══════════════════════════════════════════════════════════════════════════

_feedback_store: dict[str, FeedbackSession] = {}


def _save_session(
    feedback_response: FeedbackResponse,  # ← sekarang terima object, bukan dict
    overall_cefr: str,
    gap_skills: list[str],
) -> str:
    """
    Dipanggil oleh assess.py setelah pipeline RAG selesai.
    Menyimpan FeedbackResponse ke store dan mengembalikan session_id.
    """
    session_id = str(uuid.uuid4())
    session = FeedbackSession(
        session_id   = session_id,
        created_at   = datetime.now().isoformat(),
        overall_cefr = overall_cefr,
        gap_skills   = gap_skills,
        feedback     = feedback_response,  # ← langsung assign object
    )
    _feedback_store[session_id] = session
    logger.info(f"Sesi feedback disimpan: {session_id} (CEFR: {overall_cefr})")
    return session_id


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/feedback/history",
    response_model=list[FeedbackHistoryItem],
    summary="Ambil riwayat semua sesi feedback",
)
async def get_feedback_history():
    sessions = sorted(
        _feedback_store.values(),
        key=lambda s: s.created_at,
        reverse=True,
    )
    return [
        FeedbackHistoryItem(
            session_id   = s.session_id,
            created_at   = s.created_at,
            overall_cefr = s.overall_cefr,
            gap_skills   = s.gap_skills,
            user_rating  = s.user_rating,
        )
        for s in sessions
    ]


@router.get(
    "/feedback/{session_id}",
    response_model=FeedbackSession,
    summary="Ambil detail satu sesi feedback",
)
async def get_feedback_session(session_id: str):
    session = _feedback_store.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesi feedback '{session_id}' tidak ditemukan.",
        )
    return session


@router.post(
    "/feedback/{session_id}/rating",
    summary="Simpan rating dari pengguna",
)
async def rate_feedback(session_id: str, body: RatingRequest):
    session = _feedback_store.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesi feedback '{session_id}' tidak ditemukan.",
        )
    session.user_rating = body.rating
    session.user_notes  = body.notes
    _feedback_store[session_id] = session
    logger.info(f"Rating disimpan untuk sesi {session_id}: {body.rating}/5")
    return {
        "session_id": session_id,
        "rating":     body.rating,
        "notes":      body.notes,
        "message":    "Rating berhasil disimpan.",
    }


@router.delete(
    "/feedback/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Hapus satu sesi feedback",
)
async def delete_feedback_session(session_id: str):
    if session_id not in _feedback_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesi feedback '{session_id}' tidak ditemukan.",
        )
    del _feedback_store[session_id]
    logger.info(f"Sesi feedback dihapus: {session_id}")
    return {"session_id": session_id, "message": "Sesi berhasil dihapus."}


@router.get(
    "/feedback/stats/summary",
    summary="Statistik ringkas semua sesi feedback",
)
async def get_feedback_stats():
    if not _feedback_store:
        return {
            "total_sessions":    0,
            "cefr_distribution": {},
            "top_gap_skills":    [],
            "avg_rating":        None,
            "rated_sessions":    0,
        }

    sessions = list(_feedback_store.values())

    cefr_dist: dict[str, int] = {}
    for s in sessions:
        cefr_dist[s.overall_cefr] = cefr_dist.get(s.overall_cefr, 0) + 1

    gap_counter: dict[str, int] = {}
    for s in sessions:
        for gap in s.gap_skills:
            gap_counter[gap] = gap_counter.get(gap, 0) + 1
    top_gaps = sorted(gap_counter.items(), key=lambda x: x[1], reverse=True)

    rated   = [s for s in sessions if s.user_rating is not None]
    avg_rat = round(sum(s.user_rating for s in rated) / len(rated), 2) if rated else None

    return {
        "total_sessions":    len(sessions),
        "cefr_distribution": cefr_dist,
        "top_gap_skills":    [{"skill": k, "count": v} for k, v in top_gaps[:5]],
        "avg_rating":        avg_rat,
        "rated_sessions":    len(rated),
    }