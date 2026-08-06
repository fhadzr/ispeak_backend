-- ============================================================
-- i-Speak Knowledge Base Schema untuk Supabase
-- Jalankan di Supabase SQL Editor
-- ============================================================

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Buat tabel documents
create table if not exists documents (
  id          uuid primary key default gen_random_uuid(),
  content     text not null,
  embedding   vector(1024),        -- dimensi multilingual-e5-large
  doc_type    text not null,       -- cefr_descriptor | l1_error | vocabulary | pronunciation | learning_path
  cefr_level  text default 'all', -- A1 | A2 | B1 | B2 | C1 | C2 | all
  skill_focus text default 'general', -- accuracy | complexity | fluency | pronunciation | general
  source_doc  text not null,       -- nama file markdown sumber
  chunk_index integer default 0,   -- urutan chunk dalam dokumen
  created_at  timestamptz default now()
);

-- 3. Index untuk pencarian vektor (IVFFlat)
create index if not exists documents_embedding_idx
  on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Index untuk filter metadata
create index if not exists documents_doc_type_idx on documents (doc_type);
create index if not exists documents_cefr_level_idx on documents (cefr_level);
create index if not exists documents_skill_focus_idx on documents (skill_focus);

-- 5. RPC Function untuk similarity search
-- Dipanggil dari Python: supabase.rpc("match_documents", {...})
create or replace function match_documents (
  query_embedding vector(1024),
  match_threshold float default 0.75,
  match_count     int   default 5
)
returns table (
  id              uuid,
  content         text,
  doc_type        text,
  cefr_level      text,
  skill_focus     text,
  source_doc      text,
  similarity      float
)
language sql stable
as $$
  select
    id,
    content,
    doc_type,
    cefr_level,
    skill_focus,
    source_doc,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 6. Verifikasi setup
-- Jalankan ini untuk cek apakah schema sudah benar:
-- select count(*) from documents;
-- select routine_name from information_schema.routines where routine_name = 'match_documents';