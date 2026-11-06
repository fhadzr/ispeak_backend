"""
SQL consistency checks for the Supabase schema and RLS migration.

These tests read the .sql files and assert the security-critical properties
(RPC signature, RLS, grants). They do NOT execute SQL or touch a database.
"""

from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"

SCHEMA = (SCRIPTS / "supabase_schema.sql").read_text(encoding="utf-8")
MIGRATION = (SCRIPTS / "migrations" / "20261106_security_rls.sql").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sql_files():
    return {"schema": SCHEMA, "migration": MIGRATION}


class TestExplicitSchemaQualification:
    def test_public_qualification(self, sql_files):
        for content in sql_files.values():
            assert "public.match_documents" in content
            assert "public.documents" in content

    def test_no_public_vector_type(self, sql_files):
        # The vector extension lives in `extensions`, never `public`. No actual
        # type reference of the form `public.vector(...)` may appear.
        for content in sql_files.values():
            assert "public.vector(" not in content


class TestRpcSignature:
    def test_filter_doc_type_param(self, sql_files):
        for content in sql_files.values():
            assert "filter_doc_type text  default null" in content

    def test_security_invoker(self, sql_files):
        for content in sql_files.values():
            assert "security invoker" in content

    def test_null_filter_branch_in_where(self, sql_files):
        for content in sql_files.values():
            assert "filter_doc_type is null" in content


class TestRls:
    def test_rls_enabled(self, sql_files):
        for content in sql_files.values():
            assert "enable row level security" in content

    def test_table_access_revoked_from_public_roles(self, sql_files):
        for content in sql_files.values():
            assert "from anon" in content
            assert "from authenticated" in content
            assert "from public" in content

    def test_service_role_has_table_access(self, sql_files):
        for content in sql_files.values():
            assert "grant select" in content
            assert "to service_role" in content


class TestRpcPrivileges:
    def test_execute_revoked_from_public_roles(self, sql_files):
        for content in sql_files.values():
            assert "revoke execute on function public.match_documents" in content
            assert "from public;" in content
            assert "from anon;" in content
            assert "from authenticated;" in content

    def test_execute_granted_only_to_service_role(self, sql_files):
        for content in sql_files.values():
            assert "grant  execute on function public.match_documents" in content
            assert "to service_role;" in content

    def test_migration_drops_old_three_arg_signature(self):
        # The pre-migration function was match_documents(vector, float, int);
        # the migration must drop exactly that before recreating with 4 args.
        assert (
            "drop function if exists public.match_documents(vector(1024), float, int);"
            in MIGRATION
        )

    def test_schema_and_migration_agree_on_signature(self, sql_files):
        # Both files must define the same 4-argument signature.
        sig = "match_documents (\n  query_embedding vector(1024),\n  match_threshold float default 0.75,\n  match_count     int   default 5,\n  filter_doc_type text  default null\n)"
        assert sig in sql_files["schema"]
        assert sig in sql_files["migration"]
