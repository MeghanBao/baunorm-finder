# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Ingest the official MVV TB 2025/1 PDF
# MAGIC
# MAGIC Parses the free, official **MVV TB 2025/1** (DIBt, 354 pages) from a UC
# MAGIC Volume, chunks it into passages, and writes them to
# MAGIC `<catalog>.curated.mvvtb_chunks` in the same schema as `normen_chunks` so
# MAGIC the existing Vector Search index + agent retrieve them directly.
# MAGIC
# MAGIC These passages are **provenance='verified'** — real authoritative text
# MAGIC (unlike the synthetic examples). Legal to use: the MVV TB is published free
# MAGIC by DIBt. (Eurocode/DIN full texts are paid and NOT ingested.)

# COMMAND ----------
# MAGIC %pip install -U pypdf
# MAGIC %restart_python

# COMMAND ----------
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

for _w, _default in (
    ("env", "dev"),
    ("catalog", ""),
    ("embedding_mode", ""),
    ("pdf_path", "/Volumes/workspace/curated/raw_docs/MVVTB_2025-1.pdf"),
):
    dbutils.widgets.text(_w, _default)
os.environ["BAUNORM_ENV"] = dbutils.widgets.get("env") or "dev"
if dbutils.widgets.get("catalog"):
    os.environ["BAUNORM_CATALOG"] = dbutils.widgets.get("catalog")
if dbutils.widgets.get("embedding_mode"):
    os.environ["EMBEDDING_MODE"] = dbutils.widgets.get("embedding_mode")

from src.config import PROVENANCE_VERIFIED, config  # noqa: E402

cfg = config()
PDF_PATH = dbutils.widgets.get("pdf_path")
MVVTB_TABLE = f"{cfg.catalog}.{cfg.curated_schema}.mvvtb_chunks"
print(f"catalog={cfg.catalog}  pdf={PDF_PATH}  -> {MVVTB_TABLE}")

# COMMAND ----------
# MAGIC %md ## 1. Extract text per page

# COMMAND ----------
import re

from pypdf import PdfReader

reader = PdfReader(PDF_PATH)
pages = []
for i, page in enumerate(reader.pages, start=1):
    text = (page.extract_text() or "").strip()
    if len(text) > 60:  # skip near-empty pages
        pages.append((i, re.sub(r"[ \t]+", " ", text)))
print(f"Extracted {len(pages)} non-empty pages of {len(reader.pages)}")

# COMMAND ----------
# MAGIC %md ## 2. Chunk into passages (page + detected Anlage section)

# COMMAND ----------
import hashlib

_ANLAGE = re.compile(r"Anlage\s+A\s+[\d.\/]+")
_ZU_NORM = re.compile(r"Zu\s+(DIN[^\n]{0,70})")
CHUNK_CHARS, OVERLAP = 1100, 150


def _section(text: str) -> str:
    m = _ANLAGE.search(text)
    return m.group(0).strip() if m else ""


def _keywords(text: str) -> str:
    return ";".join(sorted({m.group(1).strip()[:60] for m in _ZU_NORM.finditer(text)}))


rows = []
for page_no, text in pages:
    section = _section(text)
    keywords = _keywords(text)
    start = 0
    while start < len(text):
        passage = text[start : start + CHUNK_CHARS].strip()
        if len(passage) > 80:
            cid = hashlib.md5(f"mvvtb|{page_no}|{start}".encode()).hexdigest()
            rows.append(
                {
                    "chunk_id": cid,
                    "norm": "MVV TB 2025/1",
                    "abschnitt": section or f"Seite {page_no}",
                    "stichworte": keywords,
                    "zusammenfassung": "",
                    "wert": "",
                    "provenance": PROVENANCE_VERIFIED,
                    "disclaimer": f"Amtlicher MVV-TB-Text (DIBt), Seite {page_no} von 354",
                    "chunk": f"MVV TB 2025/1, {section or ('Seite ' + str(page_no))}:\n{passage}",
                }
            )
        start += CHUNK_CHARS - OVERLAP

print(f"Built {len(rows)} passages")

# COMMAND ----------
# MAGIC %md ## 3. Write mvvtb_chunks (CDF-ready)

# COMMAND ----------
from pyspark.sql import Row

(
    spark.createDataFrame([Row(**r) for r in rows])
    .write.mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(MVVTB_TABLE)
)
print(f"Wrote {len(rows)} rows -> {MVVTB_TABLE}")
display(spark.sql(f"SELECT abschnitt, left(chunk, 120) AS chunk FROM {MVVTB_TABLE} LIMIT 5"))  # noqa: F821
