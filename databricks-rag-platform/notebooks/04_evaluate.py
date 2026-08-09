# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Evaluation (Mosaic AI Agent Evaluation)
# MAGIC
# MAGIC Runs the deployed agent against a German eval set with LLM-judge scorers
# MAGIC (correctness, groundedness, relevance, safety) and **gates promotion**: the
# MAGIC job fails if quality is below threshold, so a bad build isn't promoted.

# COMMAND ----------
# MAGIC %pip install -U mlflow databricks-agents
# MAGIC %restart_python

# COMMAND ----------
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

dbutils.widgets.text("env", "dev")
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env")

from src.config import config  # noqa: E402

cfg = config()

# Quality gates — tune per environment. Mean judge scores are on a 0..1 scale.
THRESHOLDS = {"correctness": 0.70, "groundedness": 0.80, "relevance": 0.75}

# COMMAND ----------
# MAGIC %md ## 1. Load the eval set (jsonl -> Delta for lineage) and build eval frame

# COMMAND ----------
eval_path = os.path.abspath(os.path.join(os.getcwd(), "..", "eval", "eval_set.jsonl"))
with open(eval_path, encoding="utf-8") as f:
    eval_rows = [json.loads(line) for line in f if line.strip()]

# Persist for lineage/audit.
spark.createDataFrame(eval_rows).write.mode("overwrite").saveAsTable(cfg.eval_set)

import pandas as pd  # noqa: E402

eval_df = pd.DataFrame(
    [
        {
            "request": r["request"],
            "expected_response": f"Norm: {r['expected_norm']}. {r['expected_facts']}",
        }
        for r in eval_rows
    ]
)

# COMMAND ----------
# MAGIC %md ## 2. Run Agent Evaluation against the logged agent

# COMMAND ----------
import mlflow
from src.agent import BaunormRagAgent  # noqa: E402

agent = BaunormRagAgent()


def predict_fn(request: str) -> str:
    from mlflow.types.agent import ChatAgentMessage

    resp = agent.predict([ChatAgentMessage(role="user", content=request)])
    return resp.messages[-1].content


with mlflow.start_run(run_name=f"baunorm-eval-{cfg.env}"):
    results = mlflow.evaluate(
        model=lambda df: [predict_fn(q) for q in df["request"]],
        data=eval_df,
        model_type="databricks-agent",  # Mosaic AI Agent Evaluation judges
    )
    metrics = results.metrics
    print(json.dumps({k: v for k, v in metrics.items()}, indent=2, default=str))

# COMMAND ----------
# MAGIC %md ## 3. Gate on thresholds — fail the job if quality regressed

# COMMAND ----------
def _metric(name: str) -> float:
    # Agent Evaluation reports e.g. "correctness/mean"; be tolerant of naming.
    for key in (f"{name}/mean", f"response/llm_judged/{name}/rating/mean", name):
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return float("nan")


failures = []
for name, minimum in THRESHOLDS.items():
    score = _metric(name)
    status = "OK" if score >= minimum else "FAIL"
    print(f"{status}  {name}: {score:.3f}  (min {minimum})")
    if not (score >= minimum):
        failures.append(name)

if failures:
    raise AssertionError(
        f"Quality gate failed for {failures}. Deployment not promoted. See MLflow run."
    )
print("All quality gates passed.")
