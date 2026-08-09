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
| Generation (Standard) | `databricks-meta-llama-3-1-8b-instruct` | klein/günstig; per `GENERATION_MODEL` austauschbar (z. B. `...-3-3-70b-instruct` für bessere Qualität) |
| Embeddings (Standard, managed) | `databricks-gte-large-en` | Null-Setup, `EMBEDDING_MODE=managed`, 1024-dim |
| Embeddings (Voll-Workspace) | self-hosted `intfloat/multilingual-e5-large` | mehrsprachig (Deutsch), `EMBEDDING_MODE=self_hosted` |

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

Voraussetzungen: Databricks CLI (v0.230+) mit Workspace-Auth; Unity Catalog,
Vector Search und Model Serving verfügbar.

**Serverless / Free-Edition-Workspace (Standard hier).** Ein einzelner managed
`workspace`-Katalog, keine klassischen Cluster, managed Embeddings. Der aktive
Bundle-Teil (`resources/jobs.yml` + `resources/agent_deploy.yml`) ist genau darauf
ausgelegt (`var.catalog=workspace`, serverless Tasks, `EMBEDDING_MODE=managed`):

```bash
cd databricks-rag-platform
export DATABRICKS_CONFIG_PROFILE=<dein-profil>

# 1. Retrieval-Kern: synthesize -> build_index -> rag_smoke (serverless)
databricks bundle validate -t dev
databricks bundle deploy  -t dev
databricks bundle run baunorm_pipeline -t dev

# 2. Agent als Model-Serving-Endpoint deployen (Review App + Inference Tables)
databricks bundle run baunorm_agent_deploy -t dev
```

Ergebnis (verifiziert):
- `workspace.curated.normen` / `normen_chunks` mit Provenance,
- Vector-Search-Endpoint `baunorm-vs` + Index `normen_index` (`ONLINE`),
- Agent-Serving-Endpoint `agents_workspace-ml-baunorm_rag_agent` (`READY`) inkl.
  **Review-App-Chat-URL** — beantwortet deutsche Fragen mit Provenance-Zitaten.

**Kosten/Abbau:** Vector Search + Serving werden stündlich berechnet. Stoppen mit
`databricks bundle destroy -t dev` und Löschen des `baunorm-vs`-Endpoints.

**Voller Workspace.** Für eigenen Katalog, self-hosted multilinguale Embeddings und
die Databricks-App die Definitionen unter `resources/full-workspace/` wieder in
`include:` aufnehmen und gegen einen Workspace mit den passenden Fähigkeiten
deployen.

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
