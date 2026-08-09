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

for _w, _default in (("env", "dev"), ("catalog", ""), ("embedding_mode", "")):
    dbutils.widgets.text(_w, _default)
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env") or "dev"
if dbutils.widgets.get("catalog"):
    os.environ["BAUNORM_CATALOG"] = dbutils.widgets.get("catalog")
if dbutils.widgets.get("embedding_mode"):
    os.environ["EMBEDDING_MODE"] = dbutils.widgets.get("embedding_mode")

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

# Fold in ingested document passages (e.g. the MVV TB PDF from notebook 06) if
# present, so the same index + agent retrieve them. Schema is aligned by name.
# Use try/except rather than spark.catalog.tableExists — the latter is unreliable
# with 3-level Unity Catalog names.
_mvvtb = f"{cfg.catalog}.{cfg.curated_schema}.mvvtb_chunks"
try:
    doc_chunks = spark.table(_mvvtb).select(chunks.columns)
    n_docs = doc_chunks.count()
    chunks = chunks.unionByName(doc_chunks)
    print(f"Unioned {n_docs} passages from {_mvvtb}")
except Exception as e:  # noqa: BLE001
    print(f"No document passages to union ({_mvvtb}): {str(e)[:120]}")

(
    chunks.write.mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(cfg.normen_chunks)
)
EXPECTED_ROWS = spark.table(cfg.normen_chunks).count()
print(f"Wrote {EXPECTED_ROWS} chunks -> {cfg.normen_chunks}")

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

# Does the index already exist? Decide create-vs-sync on existence alone, so a
# transient sync error never falls through to create() and hits "already exists".
try:
    vsc.get_index(VECTOR_SEARCH_ENDPOINT, cfg.normen_index)
    index_exists = True
except Exception:
    index_exists = False

if index_exists:
    try:
        vsc.get_index(VECTOR_SEARCH_ENDPOINT, cfg.normen_index).sync()
        print(f"Index exists — triggered sync: {cfg.normen_index}")
    except Exception as e:  # e.g. a sync is already running — fine, we wait below
        print(f"Sync trigger note (continuing to wait): {str(e)[:150]}")
else:
    kwargs = dict(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=cfg.normen_index,
        source_table_name=cfg.normen_chunks,
        pipeline_type="TRIGGERED",
        primary_key=CHUNK_ID_COLUMN,
        embedding_source_column=CHUNK_TEXT_COLUMN,
        embedding_model_endpoint_name=emb_endpoint,
    )
    # For a Databricks-managed embedding endpoint the dimension is inferred; only
    # pass it explicitly for a self-hosted model.
    if EMBEDDING_MODE != "managed":
        kwargs["embedding_dimension"] = EMBEDDING_DIMENSION
    vsc.create_delta_sync_index(**kwargs)
    print(f"Created Delta Sync index: {cfg.normen_index}")

# COMMAND ----------
# MAGIC %md ## 4. Wait until the index has finished its initial sync
# MAGIC So the next pipeline task (RAG smoke test) can query it immediately.

# COMMAND ----------
import time

idx = vsc.get_index(VECTOR_SEARCH_ENDPOINT, cfg.normen_index)
# Wait until the sync has actually ingested all source rows (ready can be True on
# the previous state while a TRIGGERED re-sync is still running).
for _ in range(90):  # up to ~15 min
    status = idx.describe().get("status", {})
    ready = status.get("ready", False)
    n = status.get("indexed_row_count", 0)
    print(f"index ready={ready}  indexed_rows={n}/{EXPECTED_ROWS}")
    if ready and n >= EXPECTED_ROWS:
        break
    time.sleep(10)
else:
    print("WARNING: index did not reach expected row count within the wait window.")

# COMMAND ----------
# MAGIC %md ## 5. Smoke test the index

# COMMAND ----------
from src.config import RETRIEVE_COLUMNS, TOP_K  # noqa: E402

try:
    res = idx.similarity_search(
        query_text="Was bedeutet feuerbeständig?",
        columns=RETRIEVE_COLUMNS,
        num_results=TOP_K,
    )
    for row in res.get("result", {}).get("data_array", []):
        print(row)
except Exception as e:
    print(f"Index not queryable yet (still building?): {e}")
