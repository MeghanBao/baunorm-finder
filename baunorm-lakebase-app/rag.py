"""
RAG bridge: query the deployed Mosaic AI agent serving endpoint.

Runs inside the Databricks App (or locally with the Databricks CLI configured).
Uses the MLflow deployments client so the agent's `custom_outputs` — our
citations + grounded flag — come back intact, not just the chat text.

The endpoint name defaults to `baunorm-rag-agent` and can be overridden with
AGENT_SERVING_ENDPOINT (see app.yaml), matching src/config.py in the platform.
"""

from __future__ import annotations

import os

AGENT_SERVING_ENDPOINT = os.environ.get("AGENT_SERVING_ENDPOINT", "baunorm-rag-agent")

_client = None


def _deploy_client():
    """Lazily create the MLflow Databricks deployments client (auth from the app's
    service principal in production, or the local Databricks config in dev)."""
    global _client
    if _client is None:
        from mlflow.deployments import get_deploy_client

        _client = get_deploy_client("databricks")
    return _client


def ask(question: str) -> dict:
    """Ask the RAG agent a question.

    Returns {answer, citations, grounded}. `citations` is the list the agent put
    in custom_outputs (norm/abschnitt/provenance/badge/is_synthetic), which the
    UI renders as source cards with provenance badges.
    """
    resp = _deploy_client().predict(
        endpoint=AGENT_SERVING_ENDPOINT,
        inputs={"messages": [{"role": "user", "content": question}]},
    )
    # ChatAgent response: {"messages": [...], "custom_outputs": {...}}
    messages = resp.get("messages") or []
    answer = messages[-1].get("content", "") if messages else ""
    custom = resp.get("custom_outputs") or {}
    return {
        "answer": answer,
        "citations": custom.get("citations", []),
        "grounded": custom.get("grounded", bool(answer)),
    }
