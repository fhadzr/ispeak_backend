# Security Policy

## Melaporkan kerentanan

Jangan membuka issue publik untuk kerentanan keamanan. Laporkan secara privat
kepada pemilik proyek agar dapat diperbaiki sebelum diungkapkan.

## Versi yang didukung

Hanya branch `main` terbaru yang didukung.

## Model keamanan

- **Autentikasi** — API key lewat header `X-API-Key` (aktif bila
  `API_KEY_AUTH_ENABLED=true`). Perbandingan key menggunakan
  `secrets.compare_digest` (constant-time). Key tidak pernah dicatat ke log dan
  tidak pernah disimpan di source code.
- **Otorisasi** — Sesi feedback di-tag dengan hash key pemanggil. Key reguler
  hanya melihat/mengubah sesi miliknya; key admin melihat/mengelola semua.
- **Rate limiting** — Sliding window per proses. Untuk deployment multi-worker,
  gunakan store bersama (mis. Redis) dengan mengimplementasikan
  `store=` pada `RateLimitMiddleware`.
- **CORS** — Daftar origin eksplisit (`CORS_ORIGINS`), tanpa wildcard, dan
  `allow_credentials=False` (autentikasi berbasis header, bukan cookie).
- **Upload** — Batas ukuran diterapkan saat streaming (bukan setelah dibaca
  penuh ke memori); validasi ekstensi, MIME, magic bytes, dan durasi; file
  sementara selalu dibersihkan.
- **LLM/RAG** — Transcript diperlakukan sebagai input tidak tepercaya
  (dibatasi panjangnya); `trust_remote_code` dinonaktifkan; kegagalan retrieval
  dibedakan dari hasil kosong dan tidak menghasilkan diagnosis tanpa grounding.

## Kredensial

| Kredensial | Di mana | Aturan |
|---|---|---|
| `SUPABASE_KEY` (service_role) | `.env` (backend) | Tidak pernah di frontend atau repository |
| `GROQ_API_KEY` | `.env` (backend) | Tidak pernah di log, error, atau repository |
| API key klien | `.env` (`API_KEYS`/`ADMIN_API_KEYS`) | Tidak pernah di frontend publik |

Jika ada kredensial yang pernah masuk ke Git history (meski sudah dihapus),
**rotasi** kredensial tersebut. Menghapus dari source tidak membatalkan kunci
yang pernah terekspos.

## Dependency

`python-multipart`, `starlette`, dan `pydantic-settings` dipin ke versi yang
telah menambal CVE yang relevan (lihat laporan remediasi). `scikit-learn`,
`numpy`, dan `tensorflow` terikat pada kompatibilitas model — jangan di-upgrade
tanpa pengujian prediksi.

Periksa kerentanan dependensi secara berkala:

```powershell
pip install pip-audit
pip-audit -r requirements.txt
```
