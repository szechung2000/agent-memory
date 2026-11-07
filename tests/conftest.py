"""Test-only PostgreSQL schema isolation; never drop shared public tables."""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@pytest.fixture()
def isolated_pg_uri():
    """Make a unique schema in the configured test database and clean it up."""
    uri = os.environ.get("AM_TEST_PG_URI")
    if not uri:
        pytest.skip("AM_TEST_PG_URI is not configured")
    if uri.startswith("postgresql://"):
        uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
    schema = "am_test_" + uuid.uuid4().hex[:16]
    admin = create_engine(uri, pool_pre_ping=True)
    try:
        with admin.begin() as conn:
            # Make the extension available through the public search path first.
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"))
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        scoped = make_url(uri).update_query_dict(
            {"options": f"-csearch_path={schema},public"}
        )
        yield scoped.render_as_string(hide_password=False)
    finally:
        # The target is a newly generated test schema, never public/user tables.
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()
