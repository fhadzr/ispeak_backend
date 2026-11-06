"""
Optional Supabase connectivity check.

Skipped unless SUPABASE_URL and SUPABASE_KEY are provided. Never prints key
material and never runs against live infrastructure during normal test runs.
"""

import os

import pytest


@pytest.mark.skipif(
    not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY")),
    reason="SUPABASE_URL/SUPABASE_KEY not set — skipping live Supabase check",
)
def test_supabase_connectivity():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    client = create_client(url, key)
    result = client.table("documents").select("id", count="exact").limit(1).execute()
    assert result is not None
