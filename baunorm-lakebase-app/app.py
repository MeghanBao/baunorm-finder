"""
Baunorm-Finder als Databricks App (Lakebase-Variante).

- Serves a small Flask API + search UI for German building standards (Normen)
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Lakebase is the single source of truth: data is seeded once from data.csv
  (/api/seed) and then maintained directly through the app (/api/normen POST).
  There is no external API to mirror - unlike the Massive sync reference app.
- Every table is created REPLICA IDENTITY FULL so Lakebase Change Data Feed can
  stream row changes into Unity Catalog Delta tables for downstream analytics.

Lokal starten / run locally:
    python app.py
Deploy als Databricks App über app.yaml.
"""

import csv
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import lakebase
import rag

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baunorm-app")

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data.csv"
TABLE_NAME = os.environ.get("NORMEN_TABLE_NAME", "normen")
FEEDBACK_TABLE = os.environ.get("FEEDBACK_TABLE_NAME", "baunorm_feedback")

# CSV-Spalten (data.csv) -> Postgres-Spalten. Halten die deutschen Feldnamen des
# Originalprojekts bei, damit data.csv unverändert übernommen werden kann.
_CSV_TO_DB = {
    "Norm": "norm",
    "Abschnitt": "abschnitt",
    "Stichworte": "stichworte",
    "Zusammenfassung": "zusammenfassung",
    "Wert": "wert",
    "Anmerkung": "anmerkung",
}


def ensure_normen_table() -> None:
    """Create the `normen` table (CDF-ready) if it doesn't exist yet.

    (norm, abschnitt) is the natural key: one row per standard + section, which
    lets us UPSERT on re-seed or manual edit instead of duplicating entries.
    """
    lakebase.ensure_table_with_cdf(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id BIGSERIAL PRIMARY KEY,
            norm TEXT NOT NULL,
            abschnitt TEXT NOT NULL DEFAULT '',
            stichworte TEXT NOT NULL DEFAULT '',
            zusammenfassung TEXT NOT NULL DEFAULT '',
            wert TEXT NOT NULL DEFAULT '',
            anmerkung TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (norm, abschnitt)
        )
        """,
        TABLE_NAME,
    )


def _row_to_result(r: dict) -> dict:
    """Shape a DB row into the same JSON contract the original FastAPI app used,
    so the existing frontend logic stays compatible."""
    return {
        "code": r.get("norm", ""),
        "article": r.get("abschnitt", ""),
        "keywords": r.get("stichworte", ""),
        "summary": r.get("zusammenfassung", ""),
        "value": r.get("wert", ""),
        "note": r.get("anmerkung", ""),
    }


def _upsert_normen(rows: list[dict]) -> int:
    """UPSERT a list of {norm, abschnitt, ...} dicts into Lakebase.

    ON CONFLICT (norm, abschnitt) keeps the table de-duplicated: re-seeding or
    re-saving an existing standard/section updates it in place and bumps
    updated_at (which CDF then streams downstream).
    """
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for r in rows:
                norm = (r.get("norm") or "").strip()
                if not norm:
                    continue  # Norm is required; skip empty/comment rows
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME}
                        (norm, abschnitt, stichworte, zusammenfassung, wert, anmerkung, updated_at)
                    VALUES (%(norm)s, %(abschnitt)s, %(stichworte)s, %(zusammenfassung)s,
                            %(wert)s, %(anmerkung)s, now())
                    ON CONFLICT (norm, abschnitt) DO UPDATE SET
                        stichworte = EXCLUDED.stichworte,
                        zusammenfassung = EXCLUDED.zusammenfassung,
                        wert = EXCLUDED.wert,
                        anmerkung = EXCLUDED.anmerkung,
                        updated_at = now()
                    """,
                    {
                        "norm": norm,
                        "abschnitt": (r.get("abschnitt") or "").strip(),
                        "stichworte": r.get("stichworte") or "",
                        "zusammenfassung": r.get("zusammenfassung") or "",
                        "wert": r.get("wert") or "",
                        "anmerkung": r.get("anmerkung") or "",
                    },
                )
                count += 1
            conn.commit()
    return count


def ensure_feedback_table() -> None:
    """Create the CDF-ready audit/feedback table for RAG questions.

    Every /api/ask writes one row here (query, answer, citations, latency); the
    optional thumbs rating is filled in later by /api/feedback. REPLICA IDENTITY
    FULL (via ensure_table_with_cdf) streams the whole audit trail into Unity
    Catalog for Lakehouse Monitoring — the enterprise observability layer, reusing
    the exact same CDF plumbing the `normen` table uses.
    """
    lakebase.ensure_table_with_cdf(
        f"""
        CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            answer TEXT NOT NULL DEFAULT '',
            citations JSONB NOT NULL DEFAULT '[]'::jsonb,
            grounded BOOLEAN NOT NULL DEFAULT true,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            rating SMALLINT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        FEEDBACK_TABLE,
    )


def _log_question(query: str, result: dict, latency_ms: int) -> int:
    """Insert an audit row and return its id so the UI can attach a thumbs rating."""
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {FEEDBACK_TABLE} (query, answer, citations, grounded, latency_ms)
                VALUES (%(q)s, %(a)s, %(c)s::jsonb, %(g)s, %(l)s)
                RETURNING id
                """,
                {
                    "q": query,
                    "a": result.get("answer", ""),
                    "c": json.dumps(result.get("citations", [])),
                    "g": bool(result.get("grounded", True)),
                    "l": latency_ms,
                },
            )
            new_id = cur.fetchone()["id"]
            conn.commit()
    return new_id


@app.errorhandler(Exception)
def handle_exception(err):
    """Return all unhandled errors as JSON (not an HTML error page), so the
    frontend's resp.json() never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    """Minimal search UI for looking up German building standards."""
    return render_template("index.html")


@app.route("/api/search")
def search():
    """Keyword search across norm / section / keywords / summary (case-insensitive
    substring match, mirroring the original tool's behaviour)."""
    ensure_normen_table()
    q = (request.args.get("q") or "").strip()
    if not q:
        total = lakebase.run_query(f"SELECT count(*) AS n FROM {TABLE_NAME}")[0]["n"]
        return jsonify({"total": total, "results": []})

    like = f"%{q}%"
    rows = lakebase.run_query(
        f"""
        SELECT norm, abschnitt, stichworte, zusammenfassung, wert, anmerkung
        FROM {TABLE_NAME}
        WHERE norm ILIKE %(like)s
           OR abschnitt ILIKE %(like)s
           OR stichworte ILIKE %(like)s
           OR zusammenfassung ILIKE %(like)s
        ORDER BY norm ASC
        """,
        {"like": like},
    )
    return jsonify({"total": len(rows), "results": [_row_to_result(r) for r in rows]})


@app.route("/api/ask", methods=["POST"])
def ask():
    """Semantic RAG answer: query the deployed Mosaic AI agent, return the answer
    with provenance-tagged citations, and log the exchange for audit/monitoring."""
    ensure_feedback_table()
    body = request.get_json(silent=True) or request.form.to_dict() or {}
    question = (body.get("q") or body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Feld 'q' (Frage) ist erforderlich"}), 400

    started = time.monotonic()
    result = rag.ask(question)
    latency_ms = int((time.monotonic() - started) * 1000)

    try:
        result["id"] = _log_question(question, result, latency_ms)
    except Exception:  # never let audit logging break the user-facing answer
        logger.exception("Failed to log question to feedback table")
        result["id"] = None
    result["latency_ms"] = latency_ms
    return jsonify(result)


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Attach a thumbs rating (+1 / -1) to a previously logged question."""
    ensure_feedback_table()
    body = request.get_json(silent=True) or request.form.to_dict() or {}
    fid, rating = body.get("id"), body.get("rating")
    if fid is None or rating not in (1, -1, "1", "-1"):
        return jsonify({"error": "id und rating (+1/-1) erforderlich"}), 400
    lakebase.run_write(
        f"UPDATE {FEEDBACK_TABLE} SET rating = %(r)s WHERE id = %(id)s",
        {"r": int(rating), "id": int(fid)},
    )
    return jsonify({"status": "ok"})


@app.route("/api/normen", methods=["GET"])
def list_normen():
    """List all standards currently stored in Lakebase."""
    ensure_normen_table()
    rows = lakebase.run_query(
        f"""
        SELECT norm, abschnitt, stichworte, zusammenfassung, wert, anmerkung, updated_at
        FROM {TABLE_NAME}
        ORDER BY norm ASC, abschnitt ASC
        """
    )
    return jsonify({"total": len(rows), "results": [_row_to_result(r) for r in rows]})


@app.route("/api/normen", methods=["POST"])
def upsert_norm():
    """Add or update a single standard entry directly through the app.

    Accepts either the German field names (Norm, Abschnitt, ...) or the JSON
    contract names (code, article, keywords, summary, value, note).
    """
    ensure_normen_table()
    body = request.get_json(silent=True) or request.form.to_dict() or {}

    # Accept both the German CSV field names and the frontend JSON names.
    norm = (body.get("norm") or body.get("Norm") or body.get("code") or "").strip()
    if not norm:
        return jsonify({"error": "Feld 'norm' (Norm) ist erforderlich"}), 400

    entry = {
        "norm": norm,
        "abschnitt": body.get("abschnitt") or body.get("Abschnitt") or body.get("article") or "",
        "stichworte": body.get("stichworte") or body.get("Stichworte") or body.get("keywords") or "",
        "zusammenfassung": body.get("zusammenfassung") or body.get("Zusammenfassung") or body.get("summary") or "",
        "wert": body.get("wert") or body.get("Wert") or body.get("value") or "",
        "anmerkung": body.get("anmerkung") or body.get("Anmerkung") or body.get("note") or "",
    }
    _upsert_normen([entry])
    return jsonify({"status": "ok", "entry": _row_to_result(entry)})


@app.route("/api/seed", methods=["POST"])
def seed():
    """One-time / re-runnable seed of Lakebase from the bundled data.csv.

    This replaces the reference app's external-API sync: baunorm-finder has no
    upstream API, so the CSV of officially verified standards (MVV TB 2025/1) is
    the seed, and Lakebase becomes the single source of truth from then on.
    """
    ensure_normen_table()
    if not DATA_FILE.exists():
        return jsonify({"error": f"data.csv nicht gefunden unter {DATA_FILE}"}), 404

    with open(DATA_FILE, encoding="utf-8") as f:
        rows = [
            {db_col: (raw.get(csv_col) or "") for csv_col, db_col in _CSV_TO_DB.items()}
            for raw in csv.DictReader(f)
        ]
    inserted = _upsert_normen(rows)
    return jsonify({"status": "ok", "seeded": inserted})


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=bool(os.getenv("FLASK_DEBUG")), host=host, port=port)
