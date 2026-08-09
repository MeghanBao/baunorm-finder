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
from mlflow.types.agent import ChatAgentMessage  # noqa: E402

agent = BaunormRagAgent()

# COMMAND ----------
QUESTIONS = [
    "Was bedeutet feuerbeständig?",
    "Welche Norm regelt Erdbebennachweise in Deutschland?",
    "Wo ist die Mindestbewehrung im Stahlbeton geregelt?",
]

for q in QUESTIONS:
    resp = agent.predict([ChatAgentMessage(role="user", content=q)])
    out = resp.custom_outputs or {}
    print("=" * 80)
    print("Frage:   ", q)
    print("Grounded:", out.get("grounded"))
    print("Antwort: ", resp.messages[-1].content)
    for c in out.get("citations", []):
        print(f"  · {c['badge']}  {c['norm']} {c.get('abschnitt','')}")

# COMMAND ----------
# MAGIC %md
# MAGIC Wenn oben zu jeder Frage eine Antwort mit Norm-Zitaten und einem
# MAGIC ✅/⚠-Badge erscheint, funktioniert der Retrieval-Kern (Vector Search +
# MAGIC Llama 3.3 70B) auf diesem Workspace.
