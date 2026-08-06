# Frontend Demo i-Speak

Halaman frontend sederhana untuk menguji alur i-Speak dari audio → model i-Speak → feedback adaptif.

## Cara pakai
1. Jalankan backend:
   ```powershell
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. Buka file `frontend/index.html` di browser.
3. Pilih file audio WAV/MP3.
4. Isi prompt/template tes.
5. Klik `Analisis dan Dapatkan Feedback`.

## Endpoints yang digunakan
- `POST /api/v1/analyze`
- `POST /api/v1/assess`

## Catatan
- Pastikan backend sedang berjalan.
- Frontend ini hanya demo; untuk penggunaan nyata, hosting static page harus di-configure dan domain backend diatur di CORS.
