# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Chunking + Vector Search Index
# MAGIC
# MAGIC Builds `curated.normen_chunks` (one embeddable text per norm/section, with
# MAGIC provenance carried through) and a **Delta Sync** Vector Search index that
# MAGIC embeds with a German-capable model.
# MAGIC
# MAGIC Embeddings: default = self-hosted `intfloat/multilingual-e5-large`
# MAGIC (endpoint `baunorm-e5-multilingual`); set `EMBEDDING_MODE=managed` to use
# MAGIC `databricks-gte-large-en` with zero setup.

# COMMAND ----------
# MAGIC %pip install -U databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

dbutils.widgets.text("env", "dev")
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env")

from src.config import (  # noqa: E402
    CHUNK_ID_COLUMN,
    CHUNK_TEXT_COLUMN,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODE,
    VECTOR_SEARCH_ENDPOINT,
    config,
    embedding_endpoint,
)

cfg = config()
print(f"env={cfg.env}  embeddings={EMBEDDING_MODE} ({embedding_endpoint()})")

# COMMAND ----------
# MAGIC %md ## 1. Build the chunk table (CDF on — required for Delta Sync)
# MAGIC One chunk per row keeps citations precise (norm + section map 1:1 to a chunk).

# COMMAND ----------
from pyspark.sql import functions as F

chunks = (
    spark.table(cfg.normen)
    .withColumn(
        CHUNK_TEXT_COLUMN,
        F.concat_ws(
            "\n",
            F.concat(F.lit("Norm: "), F.col("norm")),
            F.concat(F.lit("Abschnitt: "), F.col("abschnitt")),
            F.concat(F.lit("Stichworte: "), F.col("stichworte")),
            F.concat(F.lit("Zusammenfassung: "), F.col("zusammenfassung")),
            F.concat(F.lit("Wert: "), F.col("wert")),
        ),
    )
    # Stable surrogate id — Vector Search needs a non-null primary key column.
    .withColumn(CHUNK_ID_COLUMN, F.md5(F.concat_ws("|", "norm", "abschnitt", "zusammenfassung")))
    .select(
        CHUNK_ID_COLUMN, "norm", "abschnitt", "stichworte",
        "zusammenfassung", "wert", "provenance", "disclaimer", CHUNK_TEXT_COLUMN,
    )
)

(
    chunks.write.mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(cfg.normen_chunks)
)
print(f"Wrote {chunks.count()} chunks -> {cfg.normen_chunks}")

# COMMAND ----------
# MAGIC %md ## 2. Ensure the embedding endpoint (self-hosted multilingual-e5)
# MAGIC Skipped when EMBEDDING_MODE=managed. In self_hosted mode the endpoint is
# MAGIC created by the bundle (resources/serving.yml) after the model is registered.

# COMMAND ----------
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
emb_endpoint = embedding_endpoint()

if EMBEDDING_MODE == "self_hosted":
    existing = [e.name for e in w.serving_endpoints.list()]
    if emb_endpoint not in existing:
        print(
            f"NOTE: self-hosted endpoint '{emb_endpoint}' not found. Register "
            "intfloat/multilingual-e5-large to UC and `databricks bundle deploy` "
            "resources/serving.yml, or set EMBEDDING_MODE=managed."
        )
    else:
        print(f"Embedding endpoint ready: {emb_endpoint}")
else:
    print(f"Using managed embedding endpoint: {emb_endpoint}")

# COMMAND ----------
# MAGIC %md ## 3. Create the Vector Search endpoint + Delta Sync index

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# Endpoint (idempotent).
endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VECTOR_SEARCH_ENDPOINT not in endpoints:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")
    print(f"Creating endpoint {VECTOR_SEARCH_ENDPOINT} …")
vsc.wait_for_endpoint(VECTOR_SEARCH_ENDPOINT, verbose=True)

# Delta Sync index with managed embeddings computed by our embedding endpoint.
try:
    index = vsc.get_index(VECTOR_SEARCH_ENDPOINT, cfg.normen_index)
    index.sync()
    print(f"Index exists — triggered sync: {cfg.normen_index}")
except Exception:
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=cfg.normen_index,
        source_table_name=cfg.normen_chunks,
        pipeline_type="TRIGGERED",
        primary_key=CHUNK_ID_COLUMN,
        embedding_source_column=CHUNK_TEXT_COLUMN,
        embedding_model_endpoint_name=emb_endpoint,
        embedding_dimension=EMBEDDING_DIMENSION,
    )
    print(f"Created Delta Sync index: {cfg.normen_index}")

# COMMAND ----------
# MAGIC %md ## 4. Smoke test the index
# MAGIC (Run after the index reaches ONLINE — first sync can take a few minutes.)

# COMMAND ----------
from src.config import RETRIEVE_COLUMNS, TOP_K  # noqa: E402

try:
    idx = vsc.get_index(VECTOR_SEARCH_ENDPOINT, cfg.normen_index)
    res = idx.similarity_search(
        query_text="Was bedeutet feuerbeständig?",
        columns=RETRIEVE_COLUMNS,
        num_results=TOP_K,
    )
    for row in res.get("result", {}).get("data_array", []):
        print(row)
except Exception as e:
    print(f"Index not queryable yet (still building?): {e}")
