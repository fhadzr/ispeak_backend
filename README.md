# i-Speak Diagnostic API

Backend untuk analisis kemampuan *speaking* bahasa Inggris berbasis CEFR.
Menggabungkan audio upload + ekstraksi fitur ucapan, 7 model prediksi CEFR
sub-construct, agregasi CEFR dengan *confidence moderation*, retrieval-augmented
feedback via Supabase pgvector, dan pipeline LLM (Groq) untuk menghasilkan
feedback pembelajaran yang adaptif.

## Fitur utama

- `POST /api/v1/analyze` — upload audio (WAV/MP3/M4A/OGG) → 7 sub-construct CEFR.
- `POST /api/v1/assess` — `ISpeakOutput` → feedback adaptif berbasis RAG + LLM.
- `GET  /api/v1/feedback/*` — riwayat, detail, rating, dan statistik sesi feedback.
- Autentikasi API key, rate limiting, dan ownership feedback (opsional, aktif
  lewat environment).

## Instalasi

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # jika dibutuhkan oleh feature extraction
```

Salin `.env.example` ke `.env` dan isi nilai yang diperlukan (lihat bagian
Environment). **`.env` di-ignore oleh Git — jangan pernah di-commit.**

## Menjalankan

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: `GET /` dan `GET /health`.

## Environment

| Variabel | Wajib | Keterangan |
|---|---|---|
| `SUPABASE_URL` | production | URL proyek Supabase |
| `SUPABASE_KEY` | production | Key service_role — hanya di backend, jangan pernah di frontend |
| `GROQ_API_KEY` | production | Key API Groq untuk pipeline LLM |
| `APP_ENV` | — | `development` / `testing` / `production` (default: development) |
| `API_KEY_AUTH_ENABLED` | — | `true` untuk mengaktifkan autentikasi API key |
| `API_KEYS` / `ADMIN_API_KEYS` | — | JSON array atau daftar dipisah koma |
| `RATE_LIMIT_ENABLED` | — | `true` untuk mengaktifkan rate limiting |
| `CORS_ORIGINS` | — | Daftar origin yang diizinkan (koma atau JSON array) |
| `DOCS_ENABLED` | — | `true`/`false` untuk Swagger/ReDoc/OpenAPI |

Lihat `.env.example` untuk daftar lengkap beserta default-nya.

## Autentikasi

Saat `API_KEY_AUTH_ENABLED=true`, endpoint sensitif (`/analyze`, `/assess`,
`/feedback/*`) membutuhkan header:

```
X-API-Key: <key>
```

- Key reguler hanya dapat melihat/mengubah sesi feedback miliknya sendiri.
- Key admin (`ADMIN_API_KEYS`) dapat melihat/menghapus seluruh sesi.

## Knowledge base (Supabase pgvector)

1. Buat tabel dan RPC: jalankan `scripts/supabase_schema.sql` di SQL Editor.
2. (Jika sudah ada) terapkan migrasi keamanan:
   `scripts/migrations/20261106_security_rls.sql` — mengaktifkan RLS dan
   menambahkan parameter `filter_doc_type` pada `match_documents`.
3. Index dokumen: `python scripts/index_documents.py`.
4. Verifikasi retrieval: `python scripts/evaluate_retrieval.py`.

> `--reset` pada `index_documents.py` bersifat destruktif dan membutuhkan
> konfirmasi `--yes`; ia menolak target yang tampak production kecuali dengan
> `--allow-production`.

## Testing

```powershell
pytest tests/ -v
```

Test tidak memerlukan Supabase/LLM/model live (menggunakan mock dan skip guard).

## Model

Model `.keras` dan scaler `.pkl` berada di `models/` dengan checksum manifest
`models/MANIFEST.sha256` (diverifikasi sebelum deserialisasi pickle). Perhatikan
kompatibilitas: model `.keras` membutuhkan TensorFlow >= 2.16 / Keras 3, dan
scaler `.pkl` terikat pada `scikit-learn==1.6.1`.
