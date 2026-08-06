"""
POST /api/v1/assess
Input  : ISpeakOutput (KeyFeatures + 7 SubConstructScore dari i-Speak)
Output : FeedbackResponse (diagnostic feedback + learning path)
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import ISpeakOutput, FeedbackResponse, APIResponse
from app.services.diagnostic import build_diagnostic_profile
from app.models.schemas import ISpeakOutput, APIResponse
from app.services.diagnostic import build_diagnostic_profile
from app.services.rag_pipeline import (
    retrieve_all,
    format_context_for_prompt,
    get_source_list,
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
async def assess(ispeak_output: ISpeakOutput) -> APIResponse:
    try:
        kf = ispeak_output.key_features
        sc_summary = {
            sc.name: sc.cefr_level.value for sc in ispeak_output.subconstructs
        }

        logger.info(
            f"New assessment | task={ispeak_output.task_type} | "
            f"wpm={kf.wpm:.1f} | grammar_errors={kf.grammar_errors} | "
            f"subconstructs={sc_summary}"
        )

        # Step 1: Build diagnostic profile
        profile = build_diagnostic_profile(ispeak_output)

        logger.info(
            f"Profile built | overall={profile.overall_cefr.value} | "
            f"gaps={profile.gap_skills}"
        )

        # return APIResponse(
        #     success=True,
        #     data={
        #         "overall_cefr": profile.overall_cefr,
        #         "gap_skills": profile.gap_skills,
        #         "queries": profile.priority_queries,
        #         "next_target": profile.next_target_cefr,
        #     },
        # )

        # Step 2: RAG retrieval
        chunks = retrieve_all(profile)

        if not chunks:
            logger.warning(
                "Retrieval mengembalikan 0 chunks. "
                "Kemungkinan knowledge base belum diindex atau threshold terlalu tinggi."
            )
            # Lanjutkan dengan context kosong — LLM akan set insufficient_context=true
            context_str = "[Knowledge base tidak tersedia saat ini]"
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
            f"Assessment selesai | "
            f"hallucination_check={feedback.hallucination_check_passed} | "
            f"sources={len(feedback.retrieval_sources)} | "
            f"corrections={len(feedback.priority_corrections)}"
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Simpan sesi feedback ke store (feedback.py)
        # ═══════════════════════════════════════════════════════════════════════
        session_id = _save_session(
            feedback_response=feedback.model_dump(),
            overall_cefr=profile.overall_cefr.value,  # enum → string
            gap_skills=profile.gap_skills,
        )

        # Gabungkan feedback + session_id untuk response
        response_data = feedback.model_dump()
        response_data["session_id"] = session_id

        return APIResponse(success=True, data=response_data)
    
    except Exception as e:
        logger.error(f"Assessment error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
