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

dbutils.widgets.text("env", "dev")
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env")

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
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
resp = client.predict(
    endpoint=deployment.endpoint_name,
    inputs={"messages": [{"role": "user", "content": "Was bedeutet feuerbeständig?"}]},
)
print(resp)
