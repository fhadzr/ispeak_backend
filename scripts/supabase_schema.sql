-- ============================================================
-- i-Speak Knowledge Base Schema untuk Supabase
-- Jalankan di Supabase SQL Editor (fresh install).
--
-- Untuk database yang SUDAH memiliki tabel documents dan fungsi
-- match_documents versi lama, gunakan skrip migrasi:
--   scripts/migrations/20261106_security_rls.sql
-- ============================================================

-- 1. Enable pgvector extension
-- NOTE: di Supabase, ekstensi `vector` terpasang di schema `extensions` (BUKAN
-- `public`). Tipe `vector(1024)` di bawah direferensikan tanpa kualifikasi
-- schema, dan diselesaikan lewat search_path role (Supabase menyertakan
-- `extensions` pada search_path untuk postgres/service_role). Jangan tulis
-- `public.vector`. Verifikasi lokasi ekstensi dengan:
--   select extnamespace::regnamespace from pg_extension where extname = 'vector';
create extension if not exists vector;

-- 2. Buat tabel documents
create table if not exists public.documents (
  id          uuid primary key default gen_random_uuid(),
  content     text not null,
  embedding   vector(1024),          -- dimensi multilingual-e5-large (jangan diubah tanpa bukti)
  doc_type    text not null,         -- cefr_descriptor | l1_error | vocabulary | pronunciation | learning_path
  cefr_level  text default 'all',    -- A1 | A2 | B1 | B2 | C1 | C2 | all
  skill_focus text default 'general',-- accuracy | complexity | fluency | pronunciation | general
  source_doc  text not null,         -- nama file markdown sumber
  chunk_index integer default 0,     -- urutan chunk dalam dokumen
  created_at  timestamptz default now()
);

-- 3. Index untuk pencarian vektor (IVFFlat)
create index if not exists documents_embedding_idx
  on public.documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Index untuk filter metadata
create index if not exists documents_doc_type_idx on public.documents (doc_type);
create index if not exists documents_cefr_level_idx on public.documents (cefr_level);
create index if not exists documents_skill_focus_idx on public.documents (skill_focus);

-- 5. RPC Function untuk similarity search
-- Dipanggil dari Python: supabase.rpc("match_documents", {...})
-- `filter_doc_type = null` berarti cari SEMUA doc_type (tanpa filter).
create or replace function public.match_documents (
  query_embedding vector(1024),
  match_threshold float default 0.75,
  match_count     int   default 5,
  filter_doc_type text  default null
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
language sql
stable
security invoker
as $$
  select
    d.id,
    d.content,
    d.doc_type,
    d.cefr_level,
    d.skill_focus,
    d.source_doc,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  where 1 - (d.embedding <=> query_embedding) > match_threshold
    and (filter_doc_type is null or filter_doc_type = '' or d.doc_type = filter_doc_type)
  order by d.embedding <=> query_embedding
  limit match_count;
$$;

-- 6. Row Level Security + akses role
-- Knowledge base dibaca backend (service_role key), bukan langsung oleh klien
-- publik. service_role membobol RLS (bypassrls) dan diberikan akses eksplisit
-- di bawah. anon/authenticated/public TIDAK diberi akses apa pun.
alter table public.documents enable row level security;

revoke all on table public.documents from anon;
revoke all on table public.documents from authenticated;
revoke all on table public.documents from public;

-- Backend (service_role) — akses eksplisit.
grant usage   on schema public to service_role;
grant select  on table  public.documents to service_role;

-- RPC: hanya service_role yang boleh EXECUTE.
revoke execute on function public.match_documents(vector(1024), float, int, text) from public;
revoke execute on function public.match_documents(vector(1024), float, int, text) from anon;
revoke execute on function public.match_documents(vector(1024), float, int, text) from authenticated;
grant  execute on function public.match_documents(vector(1024), float, int, text) to service_role;

-- ============================================================
-- Credential model (dokumentasi)
--   anon key        : publik, tertanam di klien; RLS TIDAK memberinya apa pun.
--   service_role key: server-only; digunakan backend; TIDAK PERNAH ke frontend.
-- ============================================================

-- 7. Verifikasi setup
-- select count(*) from documents;
-- select routine_name from information_schema.routines where routine_name = 'match_documents';
