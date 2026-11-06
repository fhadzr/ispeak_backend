# Supabase RLS & RAG Pipeline — Final Review

Date: 2026-11-06
Scope: integrasi RLS, RPC `match_documents`, privilege, dan RAG pipeline.

## 1. Perubahan yang Dilakukan

### 1.1 `app/services/rag_pipeline.py`
- `detect_doc_type()` — fallback diubah dari `"general"` menjadi `None`. Query yang
  tidak dikenal kini mencari **semua** doc_type (tanpa filter).
- `retrieve_for_query()` —
  - `top_k`/`threshold` kini pakai pemeriksaan `is None` (bukan `or`), sehingga
    nilai eksplisit `0`/`0.0` tidak tertimpa default.
  - Seluruh jalur (embedding, koneksi Supabase, RPC) dibungkus satu `try` dan
    menghasilkan `RetrievalError` yang konsisten.
  - Log hanya mencatat `type(e).__name__` dan query, **bukan** teks error mentah
    (yang bisa memuat URL/credential).
  - Pesan `RetrievalError` dibuat generik ("Knowledge base query failed.").
- `retrieve_all()` — filter klien diubah ke `doc_type is not None` (bukan
  `!= "general"`), sehingga doc_type `None` tidak membuang chunk apa pun.

### 1.2 `scripts/supabase_schema.sql`
- Seluruh referensi dikualifikasi eksplisit ke `public.documents` dan
  `public.match_documents`.
- RPC `match_documents` diberi parameter `filter_doc_type text default null`,
  `security invoker`, dan klausa `filter_doc_type is null` (cari semua tipe).
- RLS diaktifkan; `anon`/`authenticated`/`public` dicabut dari tabel.
- `service_role` diberi `usage` schema + `select` tabel secara eksplisit.
- `EXECUTE` RPC dicabut dari `public`, `anon`, `authenticated`; diberikan hanya
  ke `service_role`.
- Ditambahkan catatan lokasi ekstensi `vector` (schema `extensions`, bukan
  `public`) + query verifikasi.

### 1.3 `scripts/migrations/20261106_security_rls.sql`
- Idempoten: `drop function if exists ... match_documents(vector(1024), float, int)`
  (signature lama 3 argumen) sebelum `create or replace` versi 4 argumen.
- `revoke execute ... from public/anon/authenticated`; `grant execute ... to service_role`.
- `grant usage/select` ke `service_role`.
- Catatan ekstensi vector + dokumentasi model kredensial.

### 1.4 Tests
- `tests/test_rag.py` — ditambah: routing doc_type (termasuk fallback `None`),
  parameter RPC sesuai signature, hasil kosong vs kegagalan, `is None` untuk
  `top_k`/`threshold`, dan tidak bocornya detail error.
- `tests/test_supabase_schema.py` — **baru**: konsistensi schema vs migration
  (kualifikasi `public`, signature RPC, `security invoker`, RLS, revoke/grant
  privilege, `public.vector` tidak digunakan, drop signature lama).

## 2. Hasil Test

- `pytest tests/test_rag.py tests/test_supabase_schema.py` → **37 passed**
- `pytest tests/` (penuh) → **74 passed, 1 skipped**

## 3. Prasyarat yang HARUS diverifikasi SEBELUM menjalankan migration

> Migration **belum aman untuk dijalankan** sampai prasyarat berikut diverifikasi
> di database target (saya tidak mengakses database — sesuai batasan).

1. **Lokasi ekstensi vector.** Jalankan di SQL Editor:
   `select extnamespace::regnamespace from pg_extension where extname = 'vector';`
   Harus mengembalikan `extensions` (bukan `public`). Jika ternyata `public`,
   sesuaikan catatan/type reference sebelum lanjut.
2. **Dependensi fungsi lama.** Pastikan tidak ada kode/fungsi lain yang memanggil
   `match_documents(vector, float, int)` (3 argumen). Backend `rag_pipeline.py`
   sudah memakai 4 argumen (`filter_doc_type`).
3. **Jenis key backend.** Pastikan `SUPABASE_KEY` di `.env` adalah **service_role**
   (bukan anon). anon tidak akan bisa membaca setelah RLS aktif.
4. **Fungsi lain tidak terdampak.** Migration hanya menyentuh `match_documents`
   dan `documents`; jangan menjalankan drop pada fungsi lain.
5. **Backup.** Ambil backup/export sebelum menjalankan `alter`/`drop`.

## 4. Risiko Tersisa

1. **Semantic filtering di `retrieve_all` bersifat defence-in-depth.** RPC sudah
   memfilter via `filter_doc_type`; filter klien memastikan doc_type konsisten.
   Tidak ada risiko kebocoran antar tipe selama RPC dijalankan dengan
   `security invoker` + RLS.
2. **RLS + service_role.** `service_role` membobol RLS (bypassrls) di Supabase.
   Oleh karena itu kunci service_role **wajib** hanya di backend dan tidak pernah
   di frontend/repo.
3. **Multi-role extension.** Jika ekstensi `vector` ternyata di schema lain, type
   `vector(1024)` pada signature RPC harus mengikuti search_path role yang
   menjalankan `CREATE FUNCTION`. Verifikasi langkah 1 sebelum eksekusi.

## 5. Langkah Manual (pemilik proyek)

1. Verifikasi 5 prasyarat di atas di Supabase SQL Editor.
2. Jalankan `scripts/migrations/20261106_security_rls.sql` (jika DB sudah ada),
   atau `scripts/supabase_schema.sql` (jika fresh install).
3. Konfirmasi `SUPABASE_KEY` adalah service_role; rotasi jika ragu.
4. Jalankan `python scripts/index_documents.py` ulang bila perlu, lalu
   `python scripts/evaluate_retrieval.py` untuk memastikan retrieval berfungsi
   lewat key backend.
5. Pantau log aplikasi: kegagalan retrieval kini menghasilkan `503` + log
   `error_type`, bukan diagnosis diam-diam tanpa grounding.

## 6. Status

- `detect_doc_type` fallback → **FIXED**
- Privilege RPC (revoke public/anon/authenticated, grant service_role) → **FIXED** (belum dieksekusi)
- Konsistensi schema `public.` → **FIXED**
- Validasi vector extension → **DOCUMENTED + verifikasi manual wajib**
- Error handling RAG (konsisten, tidak bocor) → **FIXED**
- `is None` untuk `top_k`/`threshold` → **FIXED**
- Test RAG + schema → **FIXED**

**Migration database: belum dijalankan dan belum aman dijalankan sampai seluruh
prasyarat pada bagian 3 diverifikasi.**
