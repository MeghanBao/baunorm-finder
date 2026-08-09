# baunorm-finder · Enterprise AI/RAG Platform on Databricks

Ein unternehmenstauglicher **RAG-Assistent für deutsche Bau-Normen**, vollständig
nativ auf Databricks gebaut und als **ein einziges Asset Bundle** deploybar.
Aufbauend auf dem [Hauptprojekt](../README.md) (Stichwortsuche) und der
[Lakebase-App](../baunorm-lakebase-app/README.md) fügt diese Ebene semantische
Suche, LLM-Antworten mit Quellenangaben, Governance und Evaluation hinzu.

> An enterprise-grade RAG assistant for German building standards (Normen), built
> fully native to Databricks and deployable as one Asset Bundle. Generation uses
> Databricks Foundation Models (no external LLM); the corpus is provenance-governed.

---

## Architektur / Architecture

```
data.csv (5 verifizierte Zeilen – source of truth)
   │  ai_query → Llama 3.3 70B (governed synthesis)
   ▼
UC Delta: <catalog>.curated.normen   (+ provenance, disclaimer)  ──CDF──┐
   │  chunk                                                              │
   ▼                                                                     │
UC Delta: <catalog>.curated.normen_chunks   (CDF on)                     │
   │  Delta Sync  (managed embeddings: multilingual-e5-large)            │
   ▼                                                                     │
Vector Search Index: <catalog>.curated.normen_index                     │
   │  retriever tool                                                     │
   ▼                                                                     │
Mosaic AI Agent  (ChatAgent: retrieve → grounded DE prompt → Llama)     │
   │  MLflow log + agents.deploy                                         │
   ▼                                                                     │
Model Serving (agent) + Review App + Inference Tables                   │
   │  /api/ask                                                           │
   ▼                                                                     │
Databricks App (Flask) ── Anfragen/Feedback → Lakebase ─────────────────┘→ UC (Monitoring)
```

Alles bleibt innerhalb von Databricks + Unity Catalog — **kein externer LLM-Aufruf**.

---

## Komponenten / Components

| Ebene | Objekt | Datei |
|---|---|---|
| Konfiguration | UC-Namen, Modell-IDs, Provenance-Vokabular | `src/config.py` |
| Korpus (Governance) | Prompt-Bau + verifiziert/synthetisch-Regeln | `src/corpus_synth.py` |
| Ingestion | `data.csv` → `curated.normen` (+provenance) | `notebooks/01_synthesize_corpus.py` |
| Index | Chunking + CDF + Vector Search Delta Sync | `notebooks/02_build_index.py` |
| Agent | ChatAgent (Retriever + Llama + Zitate) | `src/agent.py`, `notebooks/03_log_and_deploy_agent.py` |
| Evaluation | Mosaic AI Agent Evaluation, Schwellen-Gate | `notebooks/04_evaluate.py`, `eval/eval_set.jsonl` |
| App | `/api/ask` + RAG-UI + Audit-Log | `../baunorm-lakebase-app/` |
| IaC | Asset Bundle (dev/staging/prod) | `databricks.yml`, `resources/*.yml` |

---

## Foundation Models

| Zweck | Endpoint | Hinweis |
|---|---|---|
| Generation | `databricks-meta-llama-3-3-70b-instruct` | Databricks-nativ, pay-per-token; per `GENERATION_MODEL` austauschbar |
| Embeddings (Standard) | self-hosted `intfloat/multilingual-e5-large` | mehrsprachig (Deutsch), 1024-dim |
| Embeddings (Fallback) | `databricks-gte-large-en` | Null-Setup, `EMBEDDING_MODE=managed` |

---

## Governance: verifiziert vs. synthetisch

Der Korpus ist **provenance-getaggt**. Synthetische Beispiel-Einträge (per FM
erzeugt) erhalten `provenance='synthetic'`, einen Disclaimer und **niemals einen
erfundenen amtlichen Zahlenwert** — das Feld `wert` wird durch einen Marker
ersetzt (`src/corpus_synth.py::to_synthetic_record`). Zitate und die App-UI zeigen
die Herkunft als Badge (✅ verifiziert / ⚠ synthetisch). Damit bleibt die
Ehrlichkeit des Ursprungsprojekts (geprüft vs. Platzhalter) erhalten.

---

## Deployment (Runbook)

Voraussetzungen: Databricks CLI (v0.230+) mit Workspace-Auth; Serverless bzw.
Cluster-Recht, Unity Catalog, Vector Search und Model Serving aktiviert.

```bash
cd databricks-rag-platform

# 1. Bundle prüfen und ins dev-Ziel deployen
databricks bundle validate
databricks bundle deploy -t dev

# 2. Gesamte Pipeline laufen lassen: synthesize → index → deploy → evaluate
databricks bundle run baunorm_pipeline -t dev

# 3. (einmalig) Lakebase-Secret für die App setzen – siehe ../baunorm-lakebase-app
python ../baunorm-lakebase-app/setup_secrets.py
```

Nach dem Lauf:
- `<catalog>.curated.normen` enthält verifizierte + synthetische Zeilen mit Provenance,
- der Vector-Search-Index `normen_index` ist `ONLINE`,
- der Agent-Serving-Endpoint `baunorm-rag-agent` ist `READY`,
- Evaluationsmetriken liegen in MLflow (Deployment ist Schwellen-gegated).

Promotion nach `staging` / `prod`: `databricks bundle deploy -t staging` bzw. `-t prod`
(eigener, per Env-Suffix isolierter Katalog `baunorm_staging` / `baunorm`).

---

## Lokale Entwicklung / Tests

```bash
pip install -r requirements.txt
pytest tests/ -v          # prüft Governance, Prompt-Bau, Zitat-Formatierung
```

Die `src/`-Module (`config`, `corpus_synth`, `agent`) sind dependency-arm und
off-cluster testbar; Spark/Vector-Search/Serving-Orchestrierung liegt in den
Notebooks.

---

## Verifikation end-to-end

1. `pytest tests/ -v` grün.
2. `databricks bundle validate` ohne Fehler.
3. Pipeline-Job 01→02→03→04 erfolgreich; Provenance-Verteilung stimmt.
4. Frage an den Agenten, z. B. *„Was bedeutet feuerbeständig?"* → zitiert
   DIN 4102-2 mit ✅-verifiziert-Badge; ein synthetisches Thema liefert ⚠ synthetisch.
5. In der App: RAG-Box liefert Antwort + Zitate; `baunorm_feedback` und die
   Inference-Tabelle erhalten Zeilen (CDF → UC).
