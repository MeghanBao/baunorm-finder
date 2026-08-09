# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · RAG Smoke Test (in-notebook)
# MAGIC
# MAGIC Runs the full RAG path **without** a deployed serving endpoint: the same
# MAGIC `BaunormRagAgent` retrieves from the Vector Search index and generates with
# MAGIC Llama 3.3 70B, so we can verify retrieval + grounded answers + provenance
# MAGIC citations end-to-end on a serverless workspace.

# COMMAND ----------
# MAGIC %pip install -U mlflow databricks-vectorsearch databricks-sdk
# MAGIC %restart_python

# COMMAND ----------
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

for _w, _default in (("env", "dev"), ("catalog", ""), ("embedding_mode", "")):
    dbutils.widgets.text(_w, _default)
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env") or "dev"
if dbutils.widgets.get("catalog"):
    os.environ["BAUNORM_CATALOG"] = dbutils.widgets.get("catalog")
if dbutils.widgets.get("embedding_mode"):
    os.environ["EMBEDDING_MODE"] = dbutils.widgets.get("embedding_mode")

from src.agent import BaunormRagAgent  # noqa: E402
from src.config import RETRIEVE_COLUMNS, VECTOR_SEARCH_ENDPOINT, config  # noqa: E402
from mlflow.types.agent import ChatAgentMessage  # noqa: E402

agent = BaunormRagAgent()

# COMMAND ----------
# MAGIC %md ## Wait until the index actually answers queries
# MAGIC `ready=True` can briefly precede queryability across a fresh task env, so
# MAGIC poll with a trial search and retry rather than trusting status alone.

# COMMAND ----------
import time

from databricks.vector_search.client import VectorSearchClient

cfg = config()
_idx = VectorSearchClient(disable_notice=True).get_index(
    VECTOR_SEARCH_ENDPOINT, cfg.normen_index
)
for attempt in range(30):  # up to ~5 min
    try:
        _idx.similarity_search(
            query_text="test", columns=RETRIEVE_COLUMNS, num_results=1
        )
        print(f"Index queryable after {attempt} retries.")
        break
    except Exception as e:
        print(f"  index not queryable yet ({attempt}): {str(e)[:80]}")
        time.sleep(10)
else:
    raise RuntimeError("Vector index never became queryable within the wait window.")

# COMMAND ----------
QUESTIONS = [
    "Was bedeutet feuerbeständig?",
    "Welche Norm regelt Erdbebennachweise in Deutschland?",
    "Wo ist die Mindestbewehrung im Stahlbeton geregelt?",
]

import json

results = []
for q in QUESTIONS:
    resp = agent.predict([ChatAgentMessage(role="user", content=q)])
    out = resp.custom_outputs or {}
    answer = resp.messages[-1].content
    cites = [f"{c['badge']} {c['norm']} {c.get('abschnitt','')}".strip() for c in out.get("citations", [])]
    print("=" * 80)
    print("Frage:   ", q)
    print("Grounded:", out.get("grounded"))
    print("Antwort: ", answer)
    for c in cites:
        print("  ·", c)
    results.append({"frage": q, "grounded": out.get("grounded"), "antwort": answer, "zitate": cites})

# Return a structured summary so the run output is retrievable via the Jobs API.
dbutils.notebook.exit(json.dumps(results, ensure_ascii=False))

# COMMAND ----------
# MAGIC %md
# MAGIC Wenn oben zu jeder Frage eine Antwort mit Norm-Zitaten und einem
# MAGIC ✅/⚠-Badge erscheint, funktioniert der Retrieval-Kern (Vector Search +
# MAGIC Llama 3.3 70B) auf diesem Workspace.
