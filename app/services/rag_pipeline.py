"""
RAG Pipeline Service
====================
Retrieval dari Supabase pgvector menggunakan
DiagnosticProfile (dari output i-Speak) sebagai query context.
"""
import gc
import torch
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

from app.models.schemas import DiagnosticProfile, RetrievedChunk
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────
# Singleton — load model sekali saja
# ─────────────────────────────────────────────

_supabase_client: Optional[Client] = None
_embedding_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model...")
        _embedding_model = SentenceTransformer(
            "intfloat/multilingual-e5-large-instruct",
            device=_device,
            trust_remote_code=True
        )
        logger.info("Embedding model ready.")
    return _embedding_model

def unload_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        logger.info("Unloading embedding model to free VRAM...")
        del _embedding_model
        gc.collect()
        if _device == "cuda":
            torch.cuda.empty_cache()
        _embedding_model = None
        logger.info("Embedding model unloaded. VRAM freed.")

def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_key,
        )
    return _supabase_client


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Embed query dengan prefix 'query: ' — wajib untuk multilingual-e5.
    Berbeda dengan indexing yang pakai prefix 'passage: '.
    """
    model = get_embedding_model()
    embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True,
    )
    return embedding.tolist()


# ─────────────────────────────────────────────
# Query Routing (OPTIMASI BARU)
# ─────────────────────────────────────────────

def detect_doc_type(query: str) -> str:
    query = query.lower()

    if "cefr" in query:
        return "cefr_descriptor"
    elif "pronunciation" in query or "intonation" in query:
        return "pronunciation"
    elif "error" in query or "indonesian" in query:
        return "l1_error"
    elif "vocab" in query or "word" in query:
        return "vocabulary"
    elif "learning" in query or "exercise" in query:
        return "learning_path"

    return "general"


# ─────────────────────────────────────────────
# Retrieval per query
# ─────────────────────────────────────────────

def retrieve_for_query(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-K chunks dari Supabase untuk satu query."""
    top_k = top_k or settings.top_k_retrieval
    threshold = threshold or settings.similarity_threshold

    embedding = embed_query(query)
    supabase = get_supabase()
    doc_type = detect_doc_type(query)

    try:
        result = supabase.rpc(
            "match_documents",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": top_k,
                "filter_doc_type": doc_type
            }
        ).execute()

        chunks = []
        for row in result.data:
            chunks.append(RetrievedChunk(
                content=row["content"],
                source_doc=row.get("source_doc", "unknown"),
                doc_type=row.get("doc_type", "general"),
                cefr_level=row.get("cefr_level", "all"),
                skill_focus=row.get("skill_focus", "general"),
                similarity_score=round(row.get("similarity", 0.0), 4),
                query_source=query,
            ))
        return chunks

    except Exception as e:
        logger.error(f"Retrieval error — query: '{query[:50]}' | error: {e}")
        return []


# ─────────────────────────────────────────────
# Retrieve semua dari DiagnosticProfile
# ─────────────────────────────────────────────

def retrieve_all(profile: DiagnosticProfile) -> list[RetrievedChunk]:
    """
    Retrieve chunks untuk semua priority_queries dari DiagnosticProfile.
    """

    all_chunks: list[RetrievedChunk] = []
    seen_content: set[str] = set()

    for query in profile.priority_queries:
        doc_type = detect_doc_type(query)

        logger.info(f"[QUERY] {query} | type={doc_type}")

        chunks = retrieve_for_query(query)

        for chunk in chunks:
            if doc_type != "general" and chunk.doc_type != doc_type:
                continue

            fp = chunk.content[:120].strip()

            if fp not in seen_content:
                seen_content.add(fp)
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda x: x.similarity_score, reverse=True)

    max_chars = settings.max_context_tokens * 4
    total_chars = 0
    truncated: list[RetrievedChunk] = []

    for chunk in all_chunks:
        chunk_len = len(chunk.content)
        if total_chars + chunk_len <= max_chars:
            truncated.append(chunk)
            total_chars += chunk_len
        else:
            break

    logger.info(
        f"Retrieval selesai: {len(truncated)} chunks "
        f"dari {len(profile.priority_queries)} queries "
        f"(~{total_chars} chars)"
    )

    if len(truncated) < 2:
        logger.warning(
            "Hasil retrieval sangat sedikit (<2 chunks). "
            "Cek indexing / threshold."
        )

    return truncated

# ─────────────────────────────────────────────
# Format untuk prompt LLM
# ─────────────────────────────────────────────

def format_context_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks menjadi string [CONTEXT] untuk LLM prompt.
    Tiap chunk diberi header sumber agar LLM bisa sitasi.
    """
    if not chunks:
        return "[No relevant context found in knowledge base]"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[DOC {i}] "
            f"source={chunk.source_doc} | "
            f"type={chunk.doc_type} | "
            f"cefr={chunk.cefr_level} | "
            f"skill={chunk.skill_focus} | "
            f"score={chunk.similarity_score:.3f}"
        )
        parts.append(f"{header}\n{chunk.content}")

    return "\n\n---\n\n".join(parts)


def get_source_list(chunks: list[RetrievedChunk]) -> list[str]:
    """Ambil list unik source_doc dari chunks."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        if chunk.source_doc not in seen:
            seen.add(chunk.source_doc)
            sources.append(chunk.source_doc)
    return sources