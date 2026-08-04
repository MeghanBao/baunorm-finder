"""
Lakebase (Databricks-managed Postgres) connection helper.

Verbindet sich über eine einzige LAKEBASE_URL (Standard-Postgres-Connection-URL,
z. B. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
mit einer nativen Postgres-Rolle, die ein statisches, nicht ablaufendes Passwort hat.
So bleibt das Setup bei einem Secret statt fünf einzelnen Env-Variablen.

Connects using a single LAKEBASE_URL pointing at a native Postgres role with a
static, non-expiring password. Locally the URL is read from the LAKEBASE_URL env
var (see .env); in production on Databricks Apps it is fetched from the Databricks
secret scope via the SDK.
"""

import base64
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

# Lazily created Databricks client - only needed when we actually have to reach
# into a secret scope (i.e. production). Avoids importing/authing the SDK for
# purely local runs where LAKEBASE_URL is already in the environment.
_w = None


def _lakebase_url() -> str:
    """Resolve the Lakebase connection URL.

    Prefer the LAKEBASE_URL env var (local dev via .env, or an app.yaml that
    injects it directly). Otherwise fetch + base64-decode it from the Databricks
    secret scope (database/lakebase-url by default).
    """
    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        return env_url

    global _w
    if _w is None:
        from databricks.sdk import WorkspaceClient
        _w = WorkspaceClient()
    secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def ensure_table_with_cdf(create_sql: str, table_name: str) -> None:
    """Create a table (idempotent) and set REPLICA IDENTITY FULL on it.

    REPLICA IDENTITY FULL is the precondition for Lakebase Change Data Feed (CDF):
    without it Postgres only logs primary-key columns on UPDATE/DELETE, so the
    Delta history in Unity Catalog would be missing the changed column values.
    Setting it here means every table this app creates is CDF-ready by default -
    you only need to click "Start" in the Lakebase CDF UI (see README).

    Idempotent: CREATE TABLE IF NOT EXISTS + ALTER ... REPLICA IDENTITY FULL can
    both be re-run safely on every startup.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            cur.execute(f"ALTER TABLE {table_name} REPLICA IDENTITY FULL")
            conn.commit()
