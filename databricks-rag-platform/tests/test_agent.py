"""Agent: prompt building, citation formatting, and grounded-refusal behavior.

The Databricks clients are never touched — retrieval and the LLM are monkeypatched
so these run off-cluster with no credentials."""
import pytest

from src import agent as agent_mod
from src.agent import build_prompt, format_citations, format_context

DOCS = [
    {
        "chunk_id": "a",
        "norm": "DIN 4102-2",
        "abschnitt": "6.2.4",
        "chunk": "feuerbeständig = mind. 90 Min.",
        "provenance": "verified",
        "wert": "90 min",
    },
    {
        "chunk_id": "b",
        "norm": "DIN 4109-1",
        "abschnitt": "7",
        "chunk": "Schallschutz Beispiel",
        "provenance": "synthetic",
        "wert": "— synthetisch —",
    },
]


def test_build_prompt_has_system_and_context():
    msgs = build_prompt("Was bedeutet feuerbeständig?", DOCS)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # question + retrieved norms are present in the user turn
    assert "feuerbeständig" in msgs[1]["content"]
    assert "DIN 4102-2" in msgs[1]["content"]


def test_format_context_tags_provenance():
    ctx = format_context(DOCS)
    assert "✅ verifiziert" in ctx
    assert "⚠ synthetisch" in ctx


def test_format_citations_flags_synthetic():
    cites = format_citations(DOCS)
    assert cites[0]["is_synthetic"] is False
    assert cites[0]["badge"] == "✅ verifiziert"
    assert cites[1]["is_synthetic"] is True
    assert cites[1]["badge"] == "⚠ synthetisch"


def test_predict_refuses_when_no_context(monkeypatch):
    ag = agent_mod.BaunormRagAgent()
    monkeypatch.setattr(ag, "retrieve", lambda q: [])
    from mlflow.types.agent import ChatAgentMessage

    resp = ag.predict([ChatAgentMessage(role="user", content="Unbekanntes Thema?")])
    assert resp.custom_outputs["grounded"] is False
    assert resp.messages[-1].content == agent_mod.NO_CONTEXT_ANSWER


def test_predict_grounded_answer_with_citations(monkeypatch):
    ag = agent_mod.BaunormRagAgent()
    monkeypatch.setattr(ag, "retrieve", lambda q: DOCS)

    class _Msg:
        content = "feuerbeständig bedeutet mind. 90 Minuten (DIN 4102-2)."

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    class _FakeLLM:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _Completion()

    monkeypatch.setattr(ag, "_client", lambda: _FakeLLM())
    from mlflow.types.agent import ChatAgentMessage

    resp = ag.predict([ChatAgentMessage(role="user", content="feuerbeständig?")])
    assert resp.custom_outputs["grounded"] is True
    assert len(resp.custom_outputs["citations"]) == 2
    assert "90 Minuten" in resp.messages[-1].content


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
