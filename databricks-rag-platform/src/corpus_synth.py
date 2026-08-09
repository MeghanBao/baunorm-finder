"""
Corpus synthesis helpers + governance rules.

The trustworthy seed is `data.csv` (5 rows from MVV TB 2025/1). To demonstrate
retrieval at scale we synthesize additional German norm entries with a Databricks
Foundation Model — but synthesis is *governed*:

  * every synthetic row is stamped provenance="synthetic" and carries a disclaimer;
  * synthetic rows never present a fabricated authoritative value — the `wert`
    field is replaced with SYNTHETIC_VALUE_MARKER so no invented number can be
    mistaken for an official one.

This mirrors the project's existing verified-vs-placeholder discipline (see the
main README): we don't invent values and pass them off as real.

Pure-Python and dependency-light so it is unit-testable off-cluster; the Spark /
ai_query orchestration lives in notebooks/01_synthesize_corpus.py.
"""

from __future__ import annotations

import json
from typing import Any

from .config import (
    PROVENANCE_SYNTHETIC,
    PROVENANCE_VERIFIED,
    SYNTHETIC_VALUE_MARKER,
)

# Canonical corpus columns (match data.csv, lowercased to DB/Delta convention).
COLUMNS = ["norm", "abschnitt", "stichworte", "zusammenfassung", "wert", "anmerkung"]

SYNTHETIC_DISCLAIMER = (
    "Synthetisch generierter Beispieleintrag zu Demonstrationszwecken. "
    "Kein amtlicher Normwert — vor Verwendung im Originalnormtext prüfen."
)
VERIFIED_DISCLAIMER = "Aus geprüfter Quelle (MVV TB 2025/1) übernommen."


# --------------------------------------------------------------------------
# Prompt construction for the FM (Llama 3.3 70B via ai_query)
# --------------------------------------------------------------------------
SYNTH_SYSTEM = (
    "Du bist Bauingenieur:in und kennst das deutsche Normenwesen (DIN, DIN EN "
    "Eurocodes mit Nationalem Anhang, MVV TB). Erzeuge realistische, fachlich "
    "plausible Beispiel-Einträge zu deutschen Bau-Normen. Erfinde KEINE konkreten "
    "Zahlenwerte oder Bemessungsformeln — lasse das Feld 'wert' leer. Antworte "
    "ausschließlich mit einem JSON-Array."
)


def build_synth_prompt(topics: list[str], per_topic: int = 3) -> str:
    """Build the instruction sent to the FM for a batch of themes.

    Ask for structured JSON so the output parses deterministically. We explicitly
    forbid inventing numeric values — those must come from the real norm text.
    """
    schema_hint = (
        '[{"norm": "...", "abschnitt": "...", "stichworte": "a;b;c", '
        '"zusammenfassung": "...", "anmerkung": "..."}]'
    )
    themes = "\n".join(f"- {t}" for t in topics)
    return (
        f"{SYNTH_SYSTEM}\n\n"
        f"Erzeuge je Thema {per_topic} Einträge für folgende Themen:\n{themes}\n\n"
        f"Format (genau dieses JSON-Schema, Feld 'wert' NICHT ausgeben):\n{schema_hint}\n"
        "Stichworte als durch Semikolon getrennte Liste. Nur das JSON-Array zurückgeben."
    )


# --------------------------------------------------------------------------
# Governance: normalize any incoming row into a governed corpus record
# --------------------------------------------------------------------------
def _clean(value: Any) -> str:
    return ("" if value is None else str(value)).strip()


def to_verified_record(row: dict) -> dict:
    """Shape a seed (data.csv) row into a governed, verified corpus record.

    Accepts the German CSV header names (Norm, Abschnitt, ...) or already-lower
    names, so the same function handles the raw CSV and re-processed rows.
    """
    get = lambda *keys: next((row[k] for k in keys if k in row and row[k] is not None), "")
    return {
        "norm": _clean(get("norm", "Norm")),
        "abschnitt": _clean(get("abschnitt", "Abschnitt")),
        "stichworte": _clean(get("stichworte", "Stichworte")),
        "zusammenfassung": _clean(get("zusammenfassung", "Zusammenfassung")),
        "wert": _clean(get("wert", "Wert")),
        "anmerkung": _clean(get("anmerkung", "Anmerkung")),
        "provenance": PROVENANCE_VERIFIED,
        "disclaimer": VERIFIED_DISCLAIMER,
    }


def to_synthetic_record(row: dict) -> dict:
    """Shape one FM-produced row into a governed synthetic corpus record.

    The governance guarantee lives here: `wert` is always overwritten with the
    marker so no model-invented number ever reaches users as if authoritative.
    """
    return {
        "norm": _clean(row.get("norm")),
        "abschnitt": _clean(row.get("abschnitt")),
        "stichworte": _clean(row.get("stichworte")),
        "zusammenfassung": _clean(row.get("zusammenfassung")),
        "wert": SYNTHETIC_VALUE_MARKER,  # never trust a synthetic value
        "anmerkung": _clean(row.get("anmerkung")),
        "provenance": PROVENANCE_SYNTHETIC,
        "disclaimer": SYNTHETIC_DISCLAIMER,
    }


def parse_fm_json(raw: str) -> list[dict]:
    """Parse the FM response into a list of row dicts, tolerating stray prose or
    ```json fences around the array. Returns [] on unparseable output rather than
    raising, so one bad batch can't sink the whole synthesis job."""
    if not raw:
        return []
    text = raw.strip()
    if "```" in text:
        # take the content of the first fenced block, dropping a leading ```json tag
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        text = text.removeprefix("json").strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [r for r in data if isinstance(r, dict) and _clean(r.get("norm"))]


def synthetic_records_from_fm(raw: str) -> list[dict]:
    """Full path: FM text -> governed synthetic records."""
    return [to_synthetic_record(r) for r in parse_fm_json(raw)]


# Themes the synthesis notebook expands on. Kept here (not the notebook) so the
# corpus surface area is reviewable in one governed place.
DEFAULT_TOPICS = [
    "Wärmeschutz und Energieeinsparung (GEG, DIN 4108)",
    "Schallschutz im Hochbau (DIN 4109)",
    "Standsicherheit Stahlbau (DIN EN 1993 / Eurocode 3)",
    "Holzbau (DIN EN 1995 / Eurocode 5)",
    "Mauerwerksbau (DIN EN 1996 / Eurocode 6)",
    "Geotechnik und Gründungen (DIN EN 1997 / Eurocode 7)",
    "Betondeckung und Dauerhaftigkeit (DIN EN 1992)",
    "Abdichtung erdberührter Bauteile (DIN 18533)",
    "Barrierefreies Bauen (DIN 18040)",
    "Absturzsicherung und Geländer (DIN EN ISO 14122 / Landesbauordnung)",
]
