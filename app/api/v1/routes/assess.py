"""
POST /api/v1/assess
Input  : ISpeakOutput (KeyFeatures + 7 SubConstructScore dari i-Speak)
Output : FeedbackResponse (diagnostic feedback + learning path)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import Requester, require_api_key
from app.models.schemas import ISpeakOutput, APIResponse
from app.services.diagnostic import build_diagnostic_profile
from app.services.rag_pipeline import (
    RetrievalError,
    format_context_for_prompt,
    retrieve_all,
)
from app.services.llm_engine import run_full_diagnosis
from app.api.v1.routes.feedback import _save_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/assess",
    response_model=APIResponse,
    summary="Generate adaptive feedback dari output model i-Speak",
    description="""
    Menerima output lengkap dari model i-Speak (KeyFeatures + 7 SubConstructScore),
    membangun profil diagnostik, melakukan RAG retrieval dari knowledge base,
    dan menghasilkan feedback adaptif menggunakan Qwen2.5-72B-Instruct.

    **7 Sub-constructs i-Speak:**
    Fluency, Pronunciation, Prosody, Coherence_and_Cohesion,
    Topic_Relevance, Complexity, Accuracy
    """,
)
async def assess(
    ispeak_output: ISpeakOutput,
    requester: Requester = Depends(require_api_key),
) -> APIResponse:
    try:
        kf = ispeak_output.key_features
        sc_summary = {sc.name: sc.cefr_level.value for sc in ispeak_output.subconstructs}

        logger.info(
            "New assessment | task=%s | wpm=%.1f | grammar_errors=%s | subconstructs=%s",
            ispeak_output.task_type,
            kf.wpm,
            kf.grammar_errors,
            sc_summary,
        )

        # Step 1: Build diagnostic profile
        profile = build_diagnostic_profile(ispeak_output)

        logger.info(
            "Profile built | overall=%s | gaps=%s",
            profile.overall_cefr.value,
            profile.gap_skills,
        )

        # Step 2: RAG retrieval. A retrieval *failure* (connection/DB error) is
        # distinct from "no matching chunks" and must not be silently treated as
        # an empty knowledge base — we refuse to fabricate an ungrounded
        # diagnosis. retrieve_all() raises RetrievalError on failure and returns
        # [] only when there genuinely are no matches.
        chunks = retrieve_all(profile)

        if not chunks:
            context_str = "[Knowledge base tidak tersedia saat ini]"
            logger.warning("Retrieval returned 0 chunks for this profile.")
        else:
            context_str = format_context_for_prompt(chunks)

        # Step 3: LLM diagnosis
        feedback = run_full_diagnosis(
            ispeak_output=ispeak_output,
            profile=profile,
            chunks=chunks,
            context_str=context_str,
        )

        logger.info(
            "Assessment selesai | hallucination_check=%s | sources=%s | corrections=%s",
            feedback.hallucination_check_passed,
            len(feedback.retrieval_sources),
            len(feedback.priority_corrections),
        )

        # Step 4: Store session, tagged with the caller's identity.
        session_id = _save_session(
            feedback_response=feedback,
            overall_cefr=profile.overall_cefr.value,
            gap_skills=profile.gap_skills,
            owner_id=requester.client_id,
        )

        response_data = feedback.model_dump()
        response_data["session_id"] = session_id

        return APIResponse(success=True, data=response_data)

    except RetrievalError as e:
        logger.error("Retrieval failure: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base tidak dapat diakses. Coba lagi nanti.",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Assessment error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )
