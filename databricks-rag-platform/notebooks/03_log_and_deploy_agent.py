# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Log + Deploy the RAG Agent
# MAGIC
# MAGIC Logs `src/agent.py` (a ChatAgent) to MLflow **from code**, registers it to
# MAGIC Unity Catalog, and deploys it to Model Serving via `databricks.agents.deploy`
# MAGIC — which also provisions a Review App and turns on Inference Tables.

# COMMAND ----------
# MAGIC %pip install -U mlflow databricks-agents databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

# Also put src/ on the path so MLflow's log-time exec of agent.py can `import
# config` (code_paths=[src] covers the same import at serving time).
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for _w, _default in (("env", "dev"), ("catalog", ""), ("embedding_mode", "")):
    dbutils.widgets.text(_w, _default)
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env") or "dev"
if dbutils.widgets.get("catalog"):
    os.environ["BAUNORM_CATALOG"] = dbutils.widgets.get("catalog")
if dbutils.widgets.get("embedding_mode"):
    os.environ["EMBEDDING_MODE"] = dbutils.widgets.get("embedding_mode")

from src.config import (  # noqa: E402
    GENERATION_MODEL,
    VECTOR_SEARCH_ENDPOINT,
    config,
    embedding_endpoint,
)

cfg = config()

# COMMAND ----------
# MAGIC %md ## 1. Log the agent (model-from-code) with its resource dependencies
# MAGIC Declaring resources lets Model Serving mint short-lived creds for the index
# MAGIC and the FM/embedding endpoints — no PATs baked into the model.

# COMMAND ----------
import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

mlflow.set_registry_uri("databricks-uc")

resources = [
    DatabricksVectorSearchIndex(index_name=cfg.normen_index),
    DatabricksServingEndpoint(endpoint_name=GENERATION_MODEL),
    DatabricksServingEndpoint(endpoint_name=embedding_endpoint()),
]

# A tiny input example that matches the ChatAgent schema.
input_example = {
    "messages": [{"role": "user", "content": "Was bedeutet feuerbeständig?"}]
}

with mlflow.start_run(run_name=f"baunorm-rag-{cfg.env}"):
    logged = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=os.path.abspath(os.path.join(os.getcwd(), "..", "src", "agent.py")),
        code_paths=[os.path.abspath(os.path.join(os.getcwd(), "..", "src"))],
        resources=resources,
        input_example=input_example,
        pip_requirements=[
            "mlflow",
            "databricks-vectorsearch",
            "databricks-sdk",
            # get_open_ai_client() returns an openai.OpenAI client, so the serving
            # container needs the openai package too.
            "openai",
        ],
    )
print("Logged:", logged.model_uri)

# COMMAND ----------
# MAGIC %md ## 2. Register to Unity Catalog

# COMMAND ----------
registered = mlflow.register_model(model_uri=logged.model_uri, name=cfg.agent_model)
print(f"Registered {cfg.agent_model} v{registered.version}")

# COMMAND ----------
# MAGIC %md ## 3. Deploy to Model Serving (Review App + Inference Tables)
# MAGIC Guardrails/rate limits are configured on the endpoint via AI Gateway after
# MAGIC first deploy (see resources notes / README).

# COMMAND ----------
from databricks import agents

deployment = agents.deploy(
    model_name=cfg.agent_model,
    model_version=registered.version,
    scale_to_zero=True,
    tags={"project": "baunorm", "env": cfg.env},
)
print("Serving endpoint:", deployment.endpoint_name)
print("Review App URL:", deployment.review_app_url)

# COMMAND ----------
# MAGIC %md ## 4. Smoke test the live endpoint

# COMMAND ----------
import json
import time

from mlflow.deployments import get_deploy_client

from databricks.sdk import WorkspaceClient

client = get_deploy_client("databricks")
w = WorkspaceClient()

# Endpoint provisioning takes several minutes after agents.deploy returns. Poll
# the endpoint state so we fail fast on a build error instead of blindly retrying.
resp = None
for attempt in range(90):  # up to ~30 min
    ep = w.serving_endpoints.get(deployment.endpoint_name)
    cfg_update = str(getattr(ep.state, "config_update", ""))
    ready = str(getattr(ep.state, "ready", ""))
    print(f"  [{attempt}] ready={ready} config_update={cfg_update}")
    if "FAILED" in cfg_update:
        # Surface the served-entity failure message for a fast, actionable error.
        detail = ""
        for se in (ep.config.served_entities if ep.config else []) or []:
            detail = getattr(se.state, "deployment_state_message", "") or detail
        raise RuntimeError(f"Agent endpoint build failed: {detail}")
    if "READY" in ready:
        resp = client.predict(
            endpoint=deployment.endpoint_name,
            inputs={"messages": [{"role": "user", "content": "Was bedeutet feuerbeständig?"}]},
        )
        print(f"Endpoint READY after {attempt} polls.")
        break
    time.sleep(20)
else:
    raise RuntimeError("Agent serving endpoint did not become ready within the wait window.")
print(resp)

answer = ""
try:
    answer = (resp.get("messages") or [{}])[-1].get("content", "")
except Exception:
    answer = str(resp)[:500]

dbutils.notebook.exit(
    json.dumps(
        {
            "endpoint_name": deployment.endpoint_name,
            "review_app_url": getattr(deployment, "review_app_url", ""),
            "smoke_answer": answer,
        },
        ensure_ascii=False,
    )
)
