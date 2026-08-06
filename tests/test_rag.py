"""
Unit tests untuk RAG pipeline.
Tidak memerlukan koneksi Supabase — test fungsi format dan source list.
Jalankan: pytest tests/test_rag.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schemas import RetrievedChunk
from app.services.rag_pipeline import format_context_for_prompt, get_source_list


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