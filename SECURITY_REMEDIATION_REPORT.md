# Security Remediation Report — ispeak_backend

Date: 2026-11-06
Basis: Comprehensive Security Audit (2026-11-06), 21 findings.

## A. Executive Summary

Semua temuan dari audit awal telah ditindaklanjuti. Autentikasi API-key,
rate limiting, dan ownership feedback kini terpasang dan berfungsi (aktif lewat
environment). CORS diperketat, upload di-stream dengan validasi, dependency
yang rentan diperbarui, dan frontend tidak lagi merender output LLM via
`innerHTML`.

**Belum ada komit/push, belum ada migrasi DB yang dijalankan, dan Git history
tidak diubah** — sesuai batasan yang diberikan. Sejumlah tindakan tetap
memerlukan keputusan atau eksekusi manual oleh pemilik proyek (lihat bagian J/K).

Baseline test sebelum remediasi: **gagal collect** (config wajib membutuhkan
`supabase_url`/`supabase_key`). Setelah remediasi: **45 passed, 1 skipped**.

## B. Temuan Audit Awal

21 temuan: 1 Critical, 5 High, 7 Medium, 8 Low. Ringkasan: 0 kredensial bocor,
namun API berjalan tanpa autentikasi; `security.py` ada tapi tidak dipakai;
CORS wildcard + credentials; 38 kerentanan dependency; upload dibaca penuh ke
memori; tabel Supabase tanpa RLS; XSS frontend; dan `.env` ter-track.

## C. Perbaikan yang Berhasil Dilakukan

| Finding | Status | Keterangan |
|---|---|---|
| C-1 `.env` tracked | **FIXED** | `git rm --cached .env`; `.gitignore` diperluas `.env`/`.env.*` |
| H-1 no auth | **FIXED** | Middleware + dependency `require_api_key` dipasang & di-wire |
| H-2 CORS | **FIXED** | `allow_origins=settings.cors_origins`, `allow_credentials=False` |
| H-3 dep vulns | **PARTIALLY FIXED** | multipart/starlette/pydantic-settings patched; torch & sentence-transformers tersisa (lihat J) |
| H-4 upload | **FIXED** | Streaming size cap, validasi ekstensi/MIME/magic/durasi, LanguageTool singleton |
| H-5 email pribadi | **REQUIRES MANUAL ACTION** | Perlu keputusan pemilik (rewrite history) |
| M-1 `str(e)` bocor | **FIXED** | Global exception handler + request ID + pesan generik |
| M-2 XSS frontend | **FIXED** | DOM API + `textContent`, tanpa `innerHTML` untuk data dinamis |
| M-3 Supabase RLS/RPC | **FIXED** (belum dieksekusi) | Migration SQL dibuat; RPC `filter_doc_type` diselaraskan |
| M-4 prompt injection | **FIXED** | Delimiter untrusted + batas panjang transcript |
| M-5 copyright raw_docs | **FIXED** | `git rm --cached raw_docs/*`; `.gitignore` + `raw_docs/` |
| M-6 trust_remote_code | **FIXED** | Dihapus dari `rag_pipeline.py` |
| M-7 pickle deserialize | **FIXED** | `models/MANIFEST.sha256` + verifikasi sebelum `pickle.load` |
| L-1 docs/health leak | **FIXED** | Docs off di production; health minimal |
| L-2 /assess/mock | **FIXED** | Dihapus dari katalog; mock diblokir di production |
| L-3 dead settings | **FIXED** | Setting mati dihapus; `llm_backend` dikoreksi ke `groq` |
| L-4 rate limiting | **FIXED** | Middleware terpasang; limit ketat `/analyze`,`/assess`; XFF tidak dipercaya default |
| L-5 compare_digest | **FIXED** | `secrets.compare_digest` |
| L-6 deps drift/UTF-16 | **FIXED** | `requirements.txt` UTF-8, dependensi langsung dilengkapi |
| L-7 README/SECURITY | **PARTIALLY FIXED** | README + SECURITY ditambahkan; LICENSE butuh keputusan pemilik |
| L-8 destructive reset | **FIXED** | `--yes` + `--allow-production` guard |
| Baseline test gagal collect | **FIXED** | Default config aman + `validate_for_startup()` |

## D. File yang Dimodifikasi

- `app/core/config.py` — settings dirapikan, list parsing, `validate_for_startup()`
- `app/core/security.py` — auth dependency, middleware, rate limiter instance-scoped
- `app/main.py` — wire middleware, CORS, exception handler, request ID, docs off prod
- `app/api/v1/routes/analyze.py` — streaming upload + validasi + durasi
- `app/api/v1/routes/assess.py` — auth dependency, ownership, retrieval error handling
- `app/api/v1/routes/feedback.py` — ownership model (`owner_id`)
- `app/services/rag_pipeline.py` — `RetrievalError`, hapus `trust_remote_code`
- `app/services/llm_engine.py` — hardening prompt injection + prompt size guard
- `app/services/ispeak/model_inference.py` — checksum verifikasi scaler
- `app/services/ispeak/language_tool.py` — **baru** (singleton LanguageTool)
- `frontend/index.html` — render berbasis DOM (anti XSS)
- `scripts/index_documents.py` — guard operasi destruktif
- `scripts/supabase_schema.sql` — RLS + `filter_doc_type`
- `tests/test_supabase.py` — skip-guard, tidak mencetak key
- `tests/test_security.py` — **baru** (auth/authz/ratelimit/CORS/upload)
- `.env.example`, `.gitignore`, `README.md`, `SECURITY.md`, `requirements.txt`
- `models/MANIFEST.sha256` — **baru**
- `scripts/migrations/20261106_security_rls.sql` — **baru**

## E. Dependency yang Diperbarui

| Package | Sebelum | Sesudah |
|---|---|---|
| python-multipart | 0.0.12 | 0.0.32 |
| starlette | 1.0.0 | 1.7.0 |
| pydantic-settings | 2.13.1 | 2.15.0 |

`requirements.txt` ditulis ulang sebagai UTF-8 dan kini mencantumkan dependensi
langsung yang sebelumnya transitif (`requests`, `torch`, `pandas`, dll.), serta
membuang `langchain`/`langchain-community` yang tidak diimpor kode.

## F. Migration Database yang Dibuat

`scripts/migrations/20261106_security_rls.sql`:
1. `alter table documents enable row level security`
2. Revoke akses `anon`/`authenticated`/`public`
3. Recreate `match_documents` dengan parameter `filter_doc_type` (`security invoker`)
4. Grant execute ke `service_role` + dokumentasi model kredensial

**Belum dijalankan** terhadap database mana pun (sesuai instruksi).

## G. Test yang Dijalankan

`pytest tests/ -v` — 46 item (45 passed, 1 skipped). Termasuk 18 test keamanan
baru: autentikasi (missing/wrong/correct/admin key), otorisasi ownership
(non-admin tidak bisa membaca/menghapus sesi orang lain), rate limiting
(enforced, disabled default, XFF spoofing ditolak), CORS (allowed/disallowed/
wildcard-without-credentials), dan validasi upload (ekstensi non-audio, magic
bytes). Test Supabase diskip tanpa env.

## H. Hasil Test

- `pytest tests/` → **45 passed, 1 skipped**
- `pip check` → 2 warning pra-kondisi (lihat J), tidak berkaitan dengan perubahan
- `pip-audit` (pins) → **25 sisa** (dari 38), seluruhnya torch + sentence-transformers
- Secret scan (tracked + staged + source) → **0 temuan**

## I. Vulnerability yang Belum Terselesaikan

| Package | Versi | Jumlah | Alasan tidak di-upgrade |
|---|---|---|---|
| torch | 2.4.1 | 24 | Upgrade besar berisiko memutus sentence-transformers/model; perlu pengujian inferensi |
| sentence-transformers | 3.1.0 | 1 (PYSEC-2026-4164) | Fix di 5.6.0 (major) berisiko memutus loading model e5 |

## J. Risiko Residual

1. **torch 2.4.1 + sentence-transformers 3.1.0** — kerentanan tersisa; upgrade
   perlu validasi hasil embedding/prediksi terlebih dahulu.
2. **Rate limiter in-memory per proses** — pada multi-worker, limit efektif
   dikali jumlah worker. Butuh store bersama (Redis) untuk limit keras.
3. **TensorFlow mismatch** — `pip check`: `tf-keras 2.18.0` butuh
   `tensorflow>=2.18`, tapi venv saat ini berisi `tensorflow 2.11.0`. Model
   `.keras` memerlukan TF ≥ 2.16/Keras 3. Environment serving model harus
   disesuaikan (lihat K).
4. **pyiceberg/rich** — konflik transitif pra-kondisi, tidak terkait aplikasi.
5. Repo sudah **public** sejak sebelum remediasi; commit lama tetap berisi
   `raw_docs` (file lokal tidak ikut commit berikutnya, namun blob tetap ada di
   history lama sampai pemilik memutuskan rewrite).

## K. Manual Action yang Masih Membutuhkan Pemilik Proyek

1. **Jalankan migrasi RLS** `scripts/migrations/20261106_security_rls.sql` di
   Supabase SQL Editor (jangan otomatis).
2. **Set environment production** — `APP_ENV=production`,
   `API_KEY_AUTH_ENABLED=true`, `API_KEYS`/`ADMIN_API_KEYS`,
   `RATE_LIMIT_ENABLED=true`, `CORS_ORIGINS`, `DOCS_ENABLED=false`,
   `SUPABASE_KEY` (service_role), `GROQ_API_KEY`.
3. **Keputusan LICENSE** — pilih lisensi (MIT/Apache-2.0/dll.); tidak saya buat
   karena harus keputusan pemilik.
4. **Keputusan email Git history** — `azharalbaaqi1@gmail.com` di commit
   `1a54b43`; rewrite history atau biarkan (perlu keputusan; saya dilarang
   rewrite).
5. **Rotasi kredensial** jika ada yang pernah dipakai dengan nilai asli di
   repository ini (audit tidak menemukan kebocoran, tetapi rotasi = asuransi).
6. **Perbaiki environment TensorFlow** — pastikan environment serving model
   memakai `tensorflow>=2.18` agar file `.keras` dapat dimuat.
7. **Evaluasi upgrade torch/sentence-transformers** — jalankan uji prediksi
   sebelum bump versi untuk menutup sisa kerentanan.

## L. Deployment Checklist

- [ ] `.env` tidak ter-track (`git ls-files .env` kosong) — sudah, diverifikasi
- [ ] `APP_ENV=production` + validasi startup lolos (secret lengkap, mock off)
- [ ] Migrasi RLS dijalankan di Supabase
- [ ] `DOCS_ENABLED=false` di production
- [ ] `API_KEY_AUTH_ENABLED=true` + key klien/admin dibuat
- [ ] `RATE_LIMIT_ENABLED=true`
- [ ] `CORS_ORIGINS` diisi domain frontend produksi
- [ ] `GROQ_API_KEY`/`SUPABASE_KEY` diisi (bukan placeholder)
- [ ] Environment TensorFlow ≥ 2.18 untuk `.keras`
- [ ] `pip-audit` dipantau untuk torch/sentence-transformers

## Keputusan Akhir

# READY AFTER MANUAL ACTIONS

Aplikasi tidak dapat dinyatakan *production ready* penuh sampai item di bagian
K dieksekusi — terutama migrasi RLS, konfigurasi environment produksi, dan
penyelesaian environment TensorFlow untuk memuat model. Seluruh perbaikan yang
dapat dilakukan dengan aman dari sisi kode telah selesai dan diuji.
