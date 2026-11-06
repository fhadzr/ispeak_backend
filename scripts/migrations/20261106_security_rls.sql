-- ============================================================
-- i-Speak Knowledge Base — Security Migration
-- Migration id: 20261106_security_rls
--
-- Tujuan:
--   1. Aktifkan Row Level Security pada `public.documents`.
--   2. Cabut akses anon/authenticated/public pada tabel.
--   3. Buat ulang `public.match_documents` dengan parameter `filter_doc_type`
--      (menyesuaikan app/services/rag_pipeline.py) dan `security invoker`.
--   4. Cabut EXECUTE RPC dari PUBLIC/anon/authenticated; berikan hanya ke
--      service_role. service_role juga diberi akses eksplisit ke tabel.
--
-- PENTING:
--   - File ini TIDAK dijalankan otomatis. Review dan jalankan manual di
--     Supabase SQL Editor terhadap proyek yang dituju.
--   - Tidak menghapus data.
--   - service_role key hanya untuk backend; TIDAK PERNAH ke frontend/repo.
-- ============================================================

-- 0. Catatan ekstensi vector.
--    Di Supabase, ekstensi `vector` terpasang di schema `extensions` (bukan
--    `public`). Tipe `vector(1024)` direferensikan tanpa kualifikasi dan
--    diselesaikan lewat search_path role. Jangan tulis `public.vector`.
--    Verifikasi: select extnamespace::regnamespace from pg_extension where extname = 'vector';

-- 1. Restrict direct table access for the public-facing roles.
alter table public.documents enable row level security;

revoke all on table public.documents from anon;
revoke all on table public.documents from authenticated;
revoke all on table public.documents from public;

-- Backend (service_role) — akses eksplisit. service_role memiliki bypassrls,
-- tetapi grant ini mendokumentasikan dan menjamin hak baca/usage.
grant usage  on schema public to service_role;
grant select on table  public.documents to service_role;

-- 2. Buat ulang similarity-search RPC dengan `filter_doc_type`.
--    Tanda tangan LAMA (tanpa filter_doc_type) dihapus dulu:
--      match_documents(vector(1024), float, int)
--    `create or replace` di bawah menggantikan versi 4-argumen jika sudah ada.
drop function if exists public.match_documents(vector(1024), float, int);

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

-- 3. Privilege RPC: hanya service_role yang boleh EXECUTE.
--    Cabut dari PUBLIC, anon, dan authenticated secara eksplisit.
revoke execute on function public.match_documents(vector(1024), float, int, text) from public;
revoke execute on function public.match_documents(vector(1024), float, int, text) from anon;
revoke execute on function public.match_documents(vector(1024), float, int, text) from authenticated;
grant  execute on function public.match_documents(vector(1024), float, int, text) to service_role;

-- ============================================================
-- Credential model (dokumentasi)
--   anon key          : publik, tertanam di klien; RLS memberinya tidak ada.
--   authenticated key : tidak digunakan oleh aplikasi ini.
--   service_role key  : server-only, membobol RLS; hanya backend. JANGAN
--                       pernah expose ke frontend atau repository.
-- ============================================================
