"""
Document Indexing Script
========================
Membaca dokumen markdown dari knowledge_base/,
melakukan chunking + embedding, dan menyimpan ke Supabase pgvector.

Jalankan:
  python scripts/index_documents.py              # index semua file
  python scripts/index_documents.py --reset      # reset + reindex
  python scripts/index_documents.py --file cefr_descriptors.md  # index 1 file
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config dari .env
# ─────────────────────────────────────────────
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")
EMBED_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
KB_DIR        = Path(__file__).parent.parent / "knowledge_base"

# Metadata default per file
FILE_METADATA = {
    "cefr_descriptors.md":     {"doc_type": "cefr_descriptor",  "skill_focus": "general",      "cefr_level": "all"},
    "indonesian_l1_errors.md": {"doc_type": "l1_error",         "skill_focus": "pronunciation", "cefr_level": "all"},
    "oxford_wordlist.md":      {"doc_type": "vocabulary",        "skill_focus": "complexity",    "cefr_level": "all"},
    "pronunciation_rules.md":  {"doc_type": "pronunciation",     "skill_focus": "pronunciation", "cefr_level": "all"},
    "learning_paths.md":       {"doc_type": "learning_path",     "skill_focus": "general",       "cefr_level": "all"},
}


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter dari markdown.
    Format:
    ---
    doc_type: cefr_descriptor
    cefr_level: B1
    skill_focus: fluency
    ---
    """
    meta: dict = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            return meta, parts[2].strip()
    return meta, content


def chunk_text(content: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(content) if c.strip()]


def embed_passages(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """
    Embed dengan prefix 'passage: ' — wajib untuk multilingual-e5 saat indexing.
    """
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    )
    return [e.tolist() for e in embeddings]


def index_file(
    filepath: Path,
    model: SentenceTransformer,
    supabase,
    default_meta: dict,
) -> int:
    """Index satu file. Return jumlah chunk yang berhasil disimpan."""
    logger.info(f"Processing: {filepath.name}")

    content = filepath.read_text(encoding="utf-8")
    file_meta, body = parse_frontmatter(content)
    # frontmatter override default metadata
    meta = {**default_meta, **file_meta}

    chunks = chunk_text(body)
    if not chunks:
        logger.warning(f"  Tidak ada chunk dari {filepath.name}")
        return 0

    logger.info(f"  {len(chunks)} chunks akan diindex")
    embeddings = embed_passages(chunks, model)

    # Upsert ke Supabase dalam batch 50
    BATCH = 50
    total = 0

    for i in range(0, len(chunks), BATCH):
        batch_chunks = chunks[i:i+BATCH]
        batch_embeds = embeddings[i:i+BATCH]

        rows = [
            {
                "content":     chunk,
                "embedding":   emb,
                "doc_type":    meta.get("doc_type", "general"),
                "cefr_level":  meta.get("cefr_level", "all"),
                "skill_focus": meta.get("skill_focus", "general"),
                "source_doc":  filepath.name,
                "chunk_index": i + j,
            }
            for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeds))
        ]

        supabase.table("documents").insert(rows).execute()
        total += len(rows)
        logger.info(f"  Batch {i//BATCH + 1}: {len(rows)} rows inserted")

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Hapus semua data lama sebelum indexing")
    parser.add_argument("--yes", action="store_true",
                        help="Konfirmasi operasi destruktif (wajib bersama --reset)")
    parser.add_argument("--allow-production", action="store_true",
                        help="Izinkan --reset terhadap URL yang tampak production (berbahaya)")
    parser.add_argument("--file", type=str, default=None,
                        help="Index hanya file tertentu (nama file saja, bukan path)")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Set SUPABASE_URL dan SUPABASE_KEY di file .env")
        sys.exit(1)

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model     = SentenceTransformer(EMBED_MODEL)
    supabase  = create_client(SUPABASE_URL, SUPABASE_KEY)

    if args.reset:
        _looks_production = (
            "supabase.co" in SUPABASE_URL
            and not any(t in SUPABASE_URL for t in ("localhost", "127.0.0.1", "127.0.0.1"))
            and not SUPABASE_URL.endswith(".supabase.co/")
        )
        if _looks_production and not args.allow_production:
            logger.error(
                "RESET ditolak: target %s tampak seperti environment production. "
                "Gunakan --allow-production untuk memaksa.",
                SUPABASE_URL,
            )
            sys.exit(1)
        if not args.yes:
            logger.error(
                "Operasi --reset bersifat DESTRUCTIF (menghapus SEMUA baris di "
                "tabel documents). Jalankan ulang dengan --reset --yes untuk "
                "mengonfirmasi. Target: %s",
                SUPABASE_URL,
            )
            sys.exit(1)

        logger.warning(
            "RESET: Menghapus semua data di tabel documents pada %s ...",
            SUPABASE_URL,
        )
        supabase.table("documents").delete().neq(
            "id", "00000000-0000-0000-0000-000000000000"
        ).execute()
        logger.info("Tabel documents dikosongkan.")

    files = [KB_DIR / args.file] if args.file else list(KB_DIR.glob("*.md"))

    if not files:
        logger.error(f"Tidak ada file .md di: {KB_DIR}")
        sys.exit(1)

    grand_total = 0
    for filepath in sorted(files):
        if not filepath.exists():
            logger.warning(f"File tidak ditemukan: {filepath}")
            continue
        default_meta = FILE_METADATA.get(
            filepath.name,
            {"doc_type": "general", "skill_focus": "general", "cefr_level": "all"},
        )
        count = index_file(filepath, model, supabase, default_meta)
        grand_total += count
        logger.info(f"  Selesai: {count} chunks\n")

    logger.info(f"{'='*50}")
    logger.info(f"TOTAL: {grand_total} chunks berhasil diindex ke Supabase")
    logger.info("Jalankan scripts/evaluate_retrieval.py untuk verifikasi kualitas retrieval.")


if __name__ == "__main__":
    main()