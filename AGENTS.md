# AI Agent Guidance for i-Speak Backend

This repository is a Python FastAPI backend for the i-Speak diagnostic API. It combines:
- audio upload + speech feature extraction (`app/api/v1/routes/analyze.py`)
- 7 sub-construct CEFR prediction models for speaking ability (`app/services/ispeak/model_inference.py`)
- diagnostic profile building and confidence-moderated CEFR aggregation (`app/services/diagnostic.py`)
- retrieval-augmented feedback generation via Supabase vector store (`app/services/rag_pipeline.py`)
- LLM-based feedback pipeline using Ollama or Dashscope (`app/services/llm_engine.py`)
- lightweight session storage and feedback endpoints (`app/api/v1/routes/feedback.py`)

## Key entrypoints
- API app: `app.main:app`
- Main routes:
  - `app/api/v1/routes/analyze.py`
  - `app/api/v1/routes/assess.py`
  - `app/api/v1/routes/feedback.py`
- Service layers:
  - `app/services/ispeak/model_inference.py`
  - `app/services/diagnostic.py`
  - `app/services/rag_pipeline.py`
  - `app/services/llm_engine.py`
- Schemas and validation: `app/models/schemas.py`
- Knowledge base docs: `knowledge_base/*.md`
- Indexing script: `scripts/index_documents.py`

## Run / development commands
- Install dependencies into the repository virtual environment using `requirements.txt`.
- Use `.env` or `.env.example` to configure required secrets and runtime settings.
- Standard dev server:
  - `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- The API exposes:
  - `GET /` health and metadata
  - `GET /health` simple readiness
  - `POST /api/v1/analyze` audio upload → i-Speak inference
  - `POST /api/v1/assess` `ISpeakOutput` → adaptive feedback
  - `GET /api/v1/feedback/*` feedback session history and rating

## Testing
- Run the integration-style API tests:
  - `pytest tests/test_api.py -v`
- Run diagnostic and RAG unit tests:
  - `pytest tests/test_diagnostic.py -v`
  - `pytest tests/test_rag.py -v`
- `tests/test_api.py` uses mocks for Supabase and LLM calls, so it is safe without live external services.

## Environment notes
- Config is loaded from `app/core/config.py` using `.env`.
- Required environment variables include at least:
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
  - `DASHSCOPE_API_KEY` or local Ollama settings
  - `USE_LOCAL_LLM`
  - `OLLAMA_BASE_URL`
  - `EMBEDDING_MODEL`
  - `LLM_MODEL`
- Knowledge base retrieval depends on Supabase pgvector and the `scripts/index_documents.py` pipeline.

## Important conventions and caveats
- The `/assess` route expects validated `ISpeakOutput` JSON from `app/models/schemas.py`.
- `app/services/rag_pipeline.py` embeds queries with `query: ` and uses `passage: ` for document indexing.
- `app/services/llm_engine.py` currently orchestrates a sequential 4-prompt pipeline.
- Feedback storage in `app/api/v1/routes/feedback.py` is currently in-memory only; it is not persistent across process restarts.
- `docker-compose.yml` exists but appears to be placeholder / empty in this repo.

## Guidance for AI coding agents
- Prefer small, targeted changes rather than broad refactors unless the user explicitly requests architectural changes.
- Preserve existing route semantics and the `ISpeakOutput` / `FeedbackResponse` contract.
- When modifying retrieval or LLM behavior, keep the current Supabase + prompt pipeline separation intact.
- Avoid changing stateful feedback storage to persistent storage without a follow-up request; the current implementation is intentionally simple.
