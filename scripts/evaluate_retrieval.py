"""
Retrieval Evaluation Script
============================
Mengevaluasi kualitas RAG retrieval menggunakan Recall@K.
Jalankan setelah indexing selesai untuk verifikasi knowledge base.

Jalankan: python scripts/evaluate_retrieval.py
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")

# ─────────────────────────────────────────────
# Gold standard test queries
# Format: (query, expected_doc_type, expected_skill)
# ─────────────────────────────────────────────
TEST_QUERIES = [
    # CEFR descriptor queries
    ("CEFR B1 speaking fluency descriptor can-do statements", "cefr_descriptor", "general"),
    ("CEFR A2 speaking accuracy grammar scope", "cefr_descriptor", "general"),
    ("CEFR B2 speaking complexity lexical range", "cefr_descriptor", "general"),

    # L1 interference queries
    ("Indonesian English learner common pronunciation errors", "l1_error", "pronunciation"),
    ("Indonesian speaker /v/ /f/ phoneme substitution English", "l1_error", "pronunciation"),
    ("consonant final deletion Indonesian accent English", "l1_error", "pronunciation"),

    # Vocabulary queries
    ("Oxford 5000 advanced vocabulary B1 level word list", "vocabulary", "complexity"),
    ("Academic Word List AWL academic English vocabulary", "vocabulary", "complexity"),

    # Pronunciation queries
    ("English stress rules word stress patterns Indonesian learner", "pronunciation", "pronunciation"),
    ("English intonation patterns flat speech remediation", "pronunciation", "pronunciation"),
    ("connected speech English pronunciation rules", "pronunciation", "pronunciation"),

    # Learning path queries
    ("B1 to B2 speaking learning path weekly plan", "learning_path", "general"),
    ("A2 to B1 pronunciation improvement exercises weekly", "learning_path", "general"),
    ("speaking fluency drills exercises intermediate learner", "learning_path", "fluency"),

    # Diagnostic queries (gabungan)
    ("grammar error correction B1 CEFR Indonesian speaker transition B2", "cefr_descriptor", "accuracy"),
    ("silent pause reduction speaking fluency A2 exercises", "learning_path", "fluency"),
]


def embed_query(text: str, model: SentenceTransformer) -> list[float]:
    prefixed = f"query: {text}"
    return model.encode(prefixed, normalize_embeddings=True).tolist()


def retrieve(query_embedding: list[float], supabase, threshold: float = 0.5, k: int = 5) -> list[dict]:
    """Retrieve dengan threshold lebih rendah untuk evaluation."""
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": k,
        }
    ).execute()
    return result.data or []


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Set SUPABASE_URL dan SUPABASE_KEY di .env")
        sys.exit(1)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Cek total dokumen
    count_result = supabase.table("documents").select("id", count="exact").execute()
    total_docs = count_result.count
    logger.info(f"Total chunks di Supabase: {total_docs}")

    if total_docs == 0:
        logger.error("Knowledge base kosong! Jalankan scripts/index_documents.py dulu.")
        sys.exit(1)

    # Evaluasi per query
    results = []
    k_values = [1, 3, 5]
    hits = {k: 0 for k in k_values}

    for query, expected_doc_type, expected_skill in TEST_QUERIES:
        logger.info(f"\nQuery: {query[:60]}...")
        embedding = embed_query(query, model)
        chunks = retrieve(embedding, supabase, threshold=0.5, k=5)

        retrieved_types = [c.get("doc_type", "") for c in chunks]
        retrieved_scores = [round(c.get("similarity", 0), 3) for c in chunks]

        # Recall@K: apakah expected_doc_type ada di top-K?
        for k in k_values:
            top_k_types = retrieved_types[:k]
            if expected_doc_type in top_k_types:
                hits[k] += 1

        logger.info(f"  Expected: {expected_doc_type}")
        logger.info(f"  Retrieved types: {retrieved_types}")
        logger.info(f"  Similarity scores: {retrieved_scores}")

        results.append({
            "query": query,
            "expected_doc_type": expected_doc_type,
            "retrieved_doc_types": retrieved_types,
            "similarity_scores": retrieved_scores,
            "top_1_hit": expected_doc_type in retrieved_types[:1],
            "top_3_hit": expected_doc_type in retrieved_types[:3],
            "top_5_hit": expected_doc_type in retrieved_types[:5],
        })

    # Summary
    n = len(TEST_QUERIES)
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    for k in k_values:
        recall = hits[k] / n
        status = "OK" if recall >= 0.7 else "PERLU PERBAIKAN"
        print(f"Recall@{k}: {hits[k]}/{n} = {recall:.1%}  [{status}]")

    print(f"\nTotal queries: {n}")
    print(f"Total chunks di Supabase: {total_docs}")

    if hits[5] / n < 0.7:
        print("\nSaran perbaikan:")
        print("- Pastikan dokumen knowledge base cukup lengkap dan relevan")
        print("- Coba turunkan similarity_threshold ke 0.65")
        print("- Tambah variasi konten untuk doc_type yang sering missed")
    else:
        print("\nRetrieval quality: BAIK — siap digunakan untuk RAG pipeline.")

    # Simpan hasil ke JSON
    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "recall_at_1": hits[1] / n,
            "recall_at_3": hits[3] / n,
            "recall_at_5": hits[5] / n,
            "total_queries": n,
            "total_chunks": total_docs,
            "per_query_results": results,
        }, f, indent=2)
    logger.info(f"\nHasil detail disimpan ke: {output_path}")


if __name__ == "__main__":
    main()