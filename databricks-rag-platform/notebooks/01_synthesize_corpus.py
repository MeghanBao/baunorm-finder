# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Korpus-Synthese  (Corpus synthesis)
# MAGIC
# MAGIC Seeds Unity Catalog from the trustworthy `data.csv` (5 verified rows) and
# MAGIC then **governed-synthesizes** additional German norm entries with a
# MAGIC Databricks Foundation Model (`ai_query` → Llama 3.3 70B).
# MAGIC
# MAGIC Governance: synthetic rows are stamped `provenance='synthetic'` and never
# MAGIC carry a fabricated authoritative value (see `src/corpus_synth.py`).
# MAGIC
# MAGIC Output: `<catalog>.curated.normen`  (verified + synthetic, with provenance).

# COMMAND ----------
import os
import sys

# Make the bundled src/ importable regardless of where the notebook runs from.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

dbutils.widgets.text("env", "dev")
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env")

from src.config import GENERATION_MODEL, config  # noqa: E402
from src import corpus_synth as cs  # noqa: E402

cfg = config()
print(f"env={cfg.env}  catalog={cfg.catalog}  model={GENERATION_MODEL}")

# COMMAND ----------
# MAGIC %md ## 1. Seed table from the verified data.csv

# COMMAND ----------
import csv
from pathlib import Path

from pyspark.sql import Row

# data.csv sits at the repo root, two levels up from notebooks/.
seed_path = Path(os.getcwd()).resolve().parents[1] / "data.csv"
with open(seed_path, encoding="utf-8") as f:
    seed_rows = [cs.to_verified_record(r) for r in csv.DictReader(f)]

spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.curated_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.ml_schema}")

seed_df = spark.createDataFrame([Row(**r) for r in seed_rows])
seed_df.write.mode("overwrite").saveAsTable(cfg.normen_seed)
print(f"Seeded {seed_df.count()} verified rows -> {cfg.normen_seed}")

# COMMAND ----------
# MAGIC %md ## 2. Governed synthesis via ai_query (Llama 3.3 70B)
# MAGIC One FM call per theme; each returns a JSON array we parse + govern in Python.

# COMMAND ----------
from pyspark.sql import functions as F

topics = cs.DEFAULT_TOPICS
prompts_df = spark.createDataFrame(
    [(t, cs.build_synth_prompt([t], per_topic=4)) for t in topics],
    ["topic", "prompt"],
)

# ai_query runs the Databricks-native FM in-warehouse; no external calls.
raw_df = prompts_df.withColumn(
    "raw",
    F.expr(f"ai_query('{GENERATION_MODEL}', prompt)"),
)
raw_outputs = [row["raw"] for row in raw_df.select("raw").collect()]

synthetic_rows = []
for raw in raw_outputs:
    synthetic_rows.extend(cs.synthetic_records_from_fm(raw))
print(f"Synthesized {len(synthetic_rows)} governed synthetic rows")

# COMMAND ----------
# MAGIC %md ## 3. Union verified + synthetic -> curated.normen

# COMMAND ----------
all_rows = seed_rows + synthetic_rows
normen_df = (
    spark.createDataFrame([Row(**r) for r in all_rows])
    .dropDuplicates(["norm", "abschnitt", "zusammenfassung"])
    .withColumn("updated_at", F.current_timestamp())
)

(
    normen_df.write.mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")  # audit-ready
    .saveAsTable(cfg.normen)
)

display(  # noqa: F821  (Databricks builtin)
    spark.sql(
        f"SELECT provenance, count(*) AS n FROM {cfg.normen} GROUP BY provenance ORDER BY provenance"
    )
)
print(f"Wrote {normen_df.count()} rows -> {cfg.normen}")
