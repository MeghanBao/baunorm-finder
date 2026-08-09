"""
RAG agent for German building standards.

A Mosaic AI Agent Framework `ChatAgent` that:
  1. retrieves top-k chunks from the Vector Search index,
  2. builds a grounded German prompt (answer *only* from retrieved context),
  3. calls a Databricks-native Foundation Model (Llama 3.3 70B),
  4. returns the answer plus citations with a provenance badge per source.

Design goals:
  * **Grounded** — if retrieval is empty, the agent says so instead of guessing.
  * **Provenance-aware** — every citation shows ✅ verifiziert / ⚠ synthetisch, so
    a synthetic example is never mistaken for an official value.
  * **Testable** — prompt building and citation formatting are pure functions
    (`build_prompt`, `format_citations`) unit-tested in tests/test_agent.py; the
    Databricks clients are only touched inside `BaunormRagAgent.predict`.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from .config import (
    AGENT_SERVING_ENDPOINT,
    GENERATION_MODEL,
    PROVENANCE_LABELS,
    PROVENANCE_SYNTHETIC,
    RETRIEVE_COLUMNS,
    TOP_K,
    VECTOR_SEARCH_ENDPOINT,
    config,
)

SYSTEM_PROMPT = (
    "Du bist ein Assistent für deutsche Bau-Normen (DIN, DIN EN Eurocodes mit "
    "Nationalem Anhang, MVV TB). Beantworte die Frage AUSSCHLIESSLICH auf Basis "
    "des bereitgestellten Kontexts. Wenn der Kontext die Antwort nicht hergibt, "
    "sage klar, dass du es auf dieser Grundlage nicht beantworten kannst — rate "
    "nicht. Nenne die einschlägige Norm und den Abschnitt. Kennzeichne Angaben "
    "aus synthetischen Beispieldaten ausdrücklich als nicht amtlich verifiziert."
)

NO_CONTEXT_ANSWER = (
    "Dazu finde ich in den hinterlegten Normen-Daten keine belastbare Grundlage. "
    "Bitte die Frage präzisieren oder den Originalnormtext heranziehen."
)


def _provenance_badge(provenance: str) -> str:
    return PROVENANCE_LABELS.get(provenance, provenance or "?")


def format_context(docs: list[dict]) -> str:
    """Render retrieved docs into a numbered German context block for the prompt."""
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(
            f"[{i}] Norm: {d.get('norm','')} | Abschnitt: {d.get('abschnitt','')} "
            f"| Herkunft: {_provenance_badge(d.get('provenance',''))}\n"
            f"{d.get('chunk','')}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, docs: list[dict]) -> list[dict]:
    """Assemble the chat messages sent to the FM. Pure function — no I/O."""
    context = format_context(docs)
    user = (
        f"Kontext:\n{context}\n\n"
        f"Frage: {question}\n\n"
        "Antworte auf Deutsch, knapp und mit Bezug auf Norm + Abschnitt."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_citations(docs: list[dict]) -> list[dict]:
    """Shape retrieved docs into citation objects for the API/UI, incl. provenance
    badge and a flag the UI can use to visually warn on synthetic sources."""
    cites = []
    for d in docs:
        provenance = d.get("provenance", "")
        cites.append(
            {
                "norm": d.get("norm", ""),
                "abschnitt": d.get("abschnitt", ""),
                "wert": d.get("wert", ""),
                "provenance": provenance,
                "badge": _provenance_badge(provenance),
                "is_synthetic": provenance == PROVENANCE_SYNTHETIC,
            }
        )
    return cites


def _last_user_message(messages: list[ChatAgentMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content or ""
    return ""


class BaunormRagAgent(ChatAgent):
    """ChatAgent wiring Vector Search retrieval to a Databricks FM.

    Clients are created lazily so the class can be imported (and its pure helpers
    tested) without Databricks credentials present.
    """

    def __init__(self) -> None:
        self.cfg = config()
        self._vs_index = None
        self._llm = None

    # -- lazy clients -------------------------------------------------------
    def _index(self):
        if self._vs_index is None:
            from databricks.vector_search.client import VectorSearchClient

            self._vs_index = VectorSearchClient(disable_notice=True).get_index(
                VECTOR_SEARCH_ENDPOINT, self.cfg.normen_index
            )
        return self._vs_index

    def _client(self):
        if self._llm is None:
            from databricks.sdk import WorkspaceClient

            self._llm = WorkspaceClient().serving_endpoints.get_open_ai_client()
        return self._llm

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, question: str) -> list[dict]:
        """Top-k similarity search; map the Vector Search array response back to
        dicts keyed by RETRIEVE_COLUMNS."""
        res = self._index().similarity_search(
            query_text=question, columns=RETRIEVE_COLUMNS, num_results=TOP_K
        )
        rows = res.get("result", {}).get("data_array", []) or []
        # data_array rows align with RETRIEVE_COLUMNS order (+ trailing score).
        return [dict(zip(RETRIEVE_COLUMNS, row)) for row in rows]

    # -- ChatAgent contract -------------------------------------------------
    @mlflow.trace(name="baunorm_rag")
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        question = _last_user_message(messages)
        docs = self.retrieve(question) if question else []

        if not docs:
            return ChatAgentResponse(
                messages=[ChatAgentMessage(role="assistant", content=NO_CONTEXT_ANSWER)],
                custom_outputs={"citations": [], "grounded": False},
            )

        completion = self._client().chat.completions.create(
            model=GENERATION_MODEL,
            messages=build_prompt(question, docs),
            temperature=0.1,
        )
        answer = completion.choices[0].message.content

        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=answer)],
            custom_outputs={"citations": format_citations(docs), "grounded": True},
        )

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        # Non-streaming fallback: emit the full response as a single chunk.
        resp = self.predict(messages, context, custom_inputs)
        for msg in resp.messages:
            yield ChatAgentChunk(delta=msg)


# MLflow model-from-code entrypoint (notebook 03 logs this module).
AGENT = BaunormRagAgent()
mlflow.models.set_model(AGENT)

__all__ = [
    "BaunormRagAgent",
    "build_prompt",
    "format_citations",
    "format_context",
    "AGENT_SERVING_ENDPOINT",
]
