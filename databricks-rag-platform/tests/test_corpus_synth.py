"""Governance + parsing tests for corpus synthesis — the honesty guarantees."""
from src import corpus_synth as cs
from src.config import (
    PROVENANCE_SYNTHETIC,
    PROVENANCE_VERIFIED,
    SYNTHETIC_VALUE_MARKER,
)


def test_verified_record_preserves_value_and_accepts_german_headers():
    rec = cs.to_verified_record(
        {"Norm": "DIN 4102-2", "Abschnitt": "6.2.4", "Wert": "F90 = 90 min"}
    )
    assert rec["provenance"] == PROVENANCE_VERIFIED
    assert rec["norm"] == "DIN 4102-2"
    assert rec["wert"] == "F90 = 90 min"  # verified values are kept as-is


def test_synthetic_record_never_carries_a_fabricated_value():
    # Even if the FM invents a value, governance strips it.
    rec = cs.to_synthetic_record(
        {"norm": "DIN 4109-1", "abschnitt": "7", "wert": "55 dB", "zusammenfassung": "x"}
    )
    assert rec["provenance"] == PROVENANCE_SYNTHETIC
    assert rec["wert"] == SYNTHETIC_VALUE_MARKER
    assert "55 dB" not in rec["wert"]


def test_parse_fm_json_handles_fenced_and_prose():
    raw = 'Gerne:\n```json\n[{"norm":"DIN X","abschnitt":"1"}]\n```'
    rows = cs.parse_fm_json(raw)
    assert rows == [{"norm": "DIN X", "abschnitt": "1"}]


def test_parse_fm_json_returns_empty_on_garbage():
    assert cs.parse_fm_json("kein JSON hier") == []
    assert cs.parse_fm_json("") == []


def test_parse_fm_json_drops_rows_without_norm():
    raw = '[{"norm":"DIN X"},{"abschnitt":"only"}]'
    rows = cs.parse_fm_json(raw)
    assert rows == [{"norm": "DIN X"}]


def test_synthetic_records_from_fm_end_to_end():
    raw = '[{"norm":"DIN 4109","abschnitt":"7","zusammenfassung":"Schallschutz"}]'
    recs = cs.synthetic_records_from_fm(raw)
    assert len(recs) == 1
    assert recs[0]["provenance"] == PROVENANCE_SYNTHETIC
    assert recs[0]["wert"] == SYNTHETIC_VALUE_MARKER
