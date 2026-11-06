"""
Unit tests untuk RAG pipeline.
Tidak memerlukan koneksi Supabase — test fungsi format dan source list.
Jalankan: pytest tests/test_rag.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.models.schemas import RetrievedChunk
from app.services import rag_pipeline as rag
from app.services.rag_pipeline import (
    RetrievalError,
    detect_doc_type,
    format_context_for_prompt,
    get_source_list,
    retrieve_for_query,
)


def make_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content="CEFR B1 speaking fluency: learners can maintain interaction...",
            source_doc="cefr_descriptors.md",
            doc_type="cefr_descriptor",
            cefr_level="B1",
            skill_focus="fluency",
            similarity_score=0.891,
            query_source="CEFR B1 fluency speaking",
        ),
        RetrievedChunk(
            content="Indonesian speakers often produce flat intonation...",
            source_doc="indonesian_l1_errors.md",
            doc_type="l1_error",
            cefr_level="all",
            skill_focus="pronunciation",
            similarity_score=0.823,
            query_source="Indonesian accent prosody",
        ),
        RetrievedChunk(
            content="Oxford 5000 vocabulary list for B1 learners...",
            source_doc="oxford_wordlist.md",
            doc_type="vocabulary",
            cefr_level="B1",
            skill_focus="complexity",
            similarity_score=0.774,
            query_source="vocabulary B1 complexity",
        ),
    ]


class TestFormatContext:

    def test_includes_all_content(self):
        chunks = make_chunks()
        ctx = format_context_for_prompt(chunks)
        assert "CEFR B1 speaking fluency" in ctx
        assert "Indonesian speakers often produce" in ctx
        assert "Oxford 5000" in ctx

    def test_includes_doc_headers(self):
        chunks = make_chunks()
        ctx = format_context_for_prompt(chunks)
        assert "cefr_descriptors.md" in ctx
        assert "indonesian_l1_errors.md" in ctx
        assert "oxford_wordlist.md" in ctx

    def test_doc_numbered_correctly(self):
        chunks = make_chunks()
        ctx = format_context_for_prompt(chunks)
        assert "DOC 1" in ctx
        assert "DOC 2" in ctx
        assert "DOC 3" in ctx

    def test_empty_returns_placeholder(self):
        ctx = format_context_for_prompt([])
        assert "No relevant context" in ctx or "tidak tersedia" in ctx.lower() or "Knowledge base" in ctx

    def test_similarity_score_in_header(self):
        chunks = make_chunks()
        ctx = format_context_for_prompt(chunks)
        assert "0.891" in ctx


class TestGetSourceList:

    def test_returns_unique_sources(self):
        chunks = make_chunks()
        # Tambah duplikat
        chunks.append(RetrievedChunk(
            content="Another CEFR content...",
            source_doc="cefr_descriptors.md",  # duplikat
            doc_type="cefr_descriptor",
            cefr_level="B1",
            skill_focus="general",
            similarity_score=0.750,
            query_source="test",
        ))
        sources = get_source_list(chunks)
        assert len(sources) == 3  # 3 unique
        assert sources.count("cefr_descriptors.md") == 1

    def test_preserves_order(self):
        chunks = make_chunks()
        sources = get_source_list(chunks)
        assert sources[0] == "cefr_descriptors.md"
        assert sources[1] == "indonesian_l1_errors.md"

    def test_empty_returns_empty(self):
        assert get_source_list([]) == []


# ─── Document-type routing ───────────────────────────────────────────────────


class TestDetectDocType:
    def test_cefr_query(self):
        assert detect_doc_type("CEFR B1 speaking fluency descriptor") == "cefr_descriptor"

    def test_pronunciation_query(self):
        assert detect_doc_type("English pronunciation intonation practice") == "pronunciation"

    def test_vocabulary_query(self):
        assert detect_doc_type("vocabulary word list for B1") == "vocabulary"

    def test_l1_error_query(self):
        assert detect_doc_type("Indonesian learner common errors") == "l1_error"

    def test_learning_path_query(self):
        assert detect_doc_type("learning path weekly exercises") == "learning_path"

    def test_unknown_query_returns_none(self):
        # Fallback must be None (search all doc types), NOT a fake "general" type.
        assert detect_doc_type("unrelated gibberish") is None


# ─── Retrieval behaviour (mocked Supabase) ───────────────────────────────────


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeRpc:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return _FakeResult(self._data)


class _FakeSupabase:
    def __init__(self, data=None, exc=None):
        self._data = data if data is not None else []
        self._exc = exc
        self.last_rpc = None

    def rpc(self, name, params):
        self.last_rpc = (name, params)
        if self._exc is not None:
            raise self._exc
        return _FakeRpc(self._data)


class TestRetrieveForQuery:
    def _patch(self, monkeypatch, supabase, embedding=None):
        monkeypatch.setattr(rag, "embed_query", lambda q: embedding or [0.1, 0.2])
        monkeypatch.setattr(rag, "get_supabase", lambda: supabase)

    def test_unknown_query_passes_no_filter(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        retrieve_for_query("totally unrelated text")
        _, params = sup.last_rpc
        assert params["filter_doc_type"] is None

    def test_cefr_query_passes_doc_type_filter(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        retrieve_for_query("CEFR B1 fluency")
        _, params = sup.last_rpc
        assert params["filter_doc_type"] == "cefr_descriptor"

    def test_pronunciation_query_passes_doc_type_filter(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        retrieve_for_query("pronunciation intonation")
        _, params = sup.last_rpc
        assert params["filter_doc_type"] == "pronunciation"

    def test_vocabulary_query_passes_doc_type_filter(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        retrieve_for_query("vocabulary words")
        _, params = sup.last_rpc
        assert params["filter_doc_type"] == "vocabulary"

    def test_rpc_name_matches_sql(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        retrieve_for_query("anything")
        name, _ = sup.last_rpc
        assert name == "match_documents"

    def test_rpc_params_match_sql_signature(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        retrieve_for_query("anything")
        _, params = sup.last_rpc
        # Signature match_documents(query_embedding, match_threshold, match_count,
        # filter_doc_type) — these keys must be exactly what the SQL expects.
        assert set(params.keys()) == {
            "query_embedding", "match_threshold", "match_count", "filter_doc_type",
        }

    def test_empty_result_returns_empty_list(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        assert retrieve_for_query("anything") == []

    def test_falsy_topk_and_threshold_respected(self, monkeypatch):
        # Explicit 0 / 0.0 must be passed through (is None check), not silently
        # replaced by defaults.
        sup = _FakeSupabase(data=[])
        self._patch(monkeypatch, sup)
        retrieve_for_query("anything", top_k=0, threshold=0.0)
        _, params = sup.last_rpc
        assert params["match_count"] == 0
        assert params["match_threshold"] == 0.0

    def test_rpc_failure_raises_retrieval_error(self, monkeypatch):
        sup = _FakeSupabase(exc=RuntimeError("boom"))
        self._patch(monkeypatch, sup)
        with pytest.raises(RetrievalError):
            retrieve_for_query("anything")

    def test_embedding_failure_raises_retrieval_error(self, monkeypatch):
        sup = _FakeSupabase(data=[])
        monkeypatch.setattr(rag, "embed_query", lambda q: (_ for _ in ()).throw(RuntimeError("embed fail")))
        monkeypatch.setattr(rag, "get_supabase", lambda: sup)
        with pytest.raises(RetrievalError):
            retrieve_for_query("anything")

    def test_error_message_does_not_leak_internal_detail(self, monkeypatch):
        sup = _FakeSupabase(exc=RuntimeError("secret-token-xyz"))
        self._patch(monkeypatch, sup)
        with pytest.raises(RetrievalError) as exc:
            retrieve_for_query("anything")
        # The raised message must be generic; the raw exception text (which could
        # embed URLs/keys) must not leak through.
        assert "secret-token-xyz" not in str(exc.value)
        assert str(exc.value) == "Knowledge base query failed."