# Security Remediation Plan — ispeak_backend

Date: 2026-11-06
Basis: Comprehensive Security Audit (2026-11-06), 21 findings.

## Baseline (Phase 0)

- Git status: clean working tree; `.env` tracked (blob empty).
- Baseline test run: **fails to collect** — `Settings()` raises `ValidationError`
  (supabase_url / supabase_key required) because `.env` is empty, and
  `tests/test_supabase.py` runs a live client at import time.
- Dependency baseline: `pip-audit` reported 38 vulns / 8 packages. Highest
  priority: `python-multipart 0.0.12`, `starlette 1.0.0`, `pydantic-settings 2.13.1`.

## Change plan (mapped to audit findings)

| Finding | Fix | Phase |
|---|---|---|
| C-1 `.env` tracked | `git rm --cached .env` (keep local file) | 1 |
| H-5 personal email | Documented; no history rewrite (owner decision) | 1 |
| M-5 copyrighted raw_docs | `git rm --cached raw_docs/*` (keep local), add to .gitignore | 1 |
| L-6 empty files / UTF-16 reqs | Delete `requirement.txt`; rewrite `requirements.txt` UTF-8 | 1,4 |
| L-7 no README/LICENSE/SECURITY | Add README.md + SECURITY.md; LICENSE requires owner decision | 1 |
| H-1 no auth | Wire middleware + `require_api_key` dependency; protect /analyze, /assess, /feedback | 2 |
| L-4 rate limiting | Register middleware; stricter limits for /analyze,/assess; no XFF trust | 2 |
| L-5 non-constant-time key compare | `secrets.compare_digest` | 2 |
| IDOR feedback | Ownership model keyed by API-key hash; admin keys see all | 2 |
| H-2 CORS wildcard + credentials | `allow_origins=settings.cors_origins`, `allow_credentials=False` | 3 |
| M-1 `str(e)` in responses | Global exception handlers + request ID; generic client messages | 3 |
| L-3 dead settings | Remove or wire; fix `llm_backend` reporting | 3,10 |
| H-3 dependency vulns | Upgraded multipart/starlette/pydantic-settings; re-audit | 4 |
| L-6 requirements drift | Rewrite requirements.txt from working env (UTF-8) | 4 |
| H-4 upload | Stream to disk, size cap during write, extension+MIME+signature validation, duration limit, LanguageTool singleton | 5 |
| M-3 Supabase RLS + RPC | Migration SQL (not run); fix `match_documents(filter_doc_type)` | 6 |
| M-4 prompt injection | Delimit transcript as untrusted, length cap | 7 |
| M-6 trust_remote_code | Remove flag | 7 |
| LLM cost | Transcript cap, bounded retries | 7 |
| M-2 frontend XSS | Replace innerHTML with DOM APIs / textContent; no inline onclick | 8 |
| M-7 pickle | Checksum manifest + verify before load (non-breaking) | 9 |
| L-1 docs in prod, health leak | Disable docs in prod; minimal health | 10 |
| L-2 /assess/mock | Remove from catalogue; mock mode blocked in prod | 10 |
| L-8 destructive reset | `--yes` + `--allow-production` guard | 11 |
| — tests | New auth/authz/ratelimit/CORS/upload tests | 12 |

## Risk & compatibility notes

- **Model / pickle compatibility:** do not upgrade `scikit-learn`, `numpy`,
  `tensorflow` — the `.pkl` scalers and `.keras` models depend on exact versions.
- **Auth default stays OFF** (`api_key_auth_enabled=false`) so local/dev behavior
  is unchanged and existing tests keep passing; production enables it via env.
- **Feedback ownership** is only meaningful when auth is enabled; with auth off
  the store remains shared (dev mode), preserving current behavior.
- **No production migration is run.** Supabase migration SQL is delivered for
  manual review/execution.

## Execution order

1. config.py → 2. security.py → 3. main.py + routes → 4. deps → 5. upload →
6. supabase SQL + rag → 7. llm_engine → 8. frontend → 9. model manifest →
10. hardening → 11. index_documents → 12. tests → 13. validation + report.
