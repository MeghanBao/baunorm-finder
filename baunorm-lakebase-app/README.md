# baunorm-finder als Databricks App (Lakebase-Variante)

Dieselbe Stichwort-zu-Wert-Suche für deutsche Bau-Normen wie das
[Hauptprojekt](../README.md), aber als **Databricks App** gebaut:

- **Flask-Backend** liest/schreibt direkt in **Lakebase** (Databricks-managed Postgres)
- **Lakebase ist die einzige Datenquelle** – kein externer API-Sync. Einmalig aus
  `data.csv` seeden (`/api/seed`), danach über die App pflegen (`/api/normen POST`)
- **Change Data Feed (CDF)** streamt jede Zeilenänderung nach Unity Catalog
  (Delta) für die Analyse-Schicht – „App-Layer Postgres, Analyse-Layer Lakehouse"

> Architektur-Vorlage: [EcZachly/databricks-lakebase-app-day-1](https://github.com/EcZachly/databricks-lakebase-app-day-1).
> Der Unterschied: das Referenz-Repo synct eine externe **Massive API** in Lakebase;
> baunorm-finder hat keine Upstream-API, daher ist Lakebase hier die *source of
> truth* statt ein Spiegel.

## Dateien

- `app.py` – Flask: `/healthz`, `/api/search`, `/api/normen` (GET/POST), `/api/seed`
- `lakebase.py` – Postgres-Connection-Helper (single `LAKEBASE_URL`); legt Tabellen
  automatisch mit `REPLICA IDENTITY FULL` an (CDF-Voraussetzung)
- `setup_secrets.py` – einmaliges Skript, speichert die Lakebase-URL in den Secret
  Scope `database/lakebase-url`
- `app.yaml` – Databricks-App-Deployment-Config
- `templates/index.html` – schlanke Such- und Erfassungs-UI
- `data.csv` – Seed-Daten (aus MVV TB 2025/1)

## Endpoints

| Endpoint | Zweck |
|---|---|
| `GET /healthz` | Health-Check |
| `GET /` | Such-UI |
| `GET /api/search?q=` | Stichwortsuche (norm/abschnitt/stichworte/zusammenfassung, ILIKE) |
| `GET /api/normen` | Alle Einträge auflisten |
| `POST /api/normen` | Einen Eintrag anlegen/aktualisieren (UPSERT auf `(norm, abschnitt)`) |
| `POST /api/seed` | `data.csv` einmalig in Lakebase laden |

## Datenmodell

Tabelle `normen` in `databricks_postgres`:

```
id BIGSERIAL PK · norm · abschnitt · stichworte · zusammenfassung · wert
· anmerkung · updated_at · UNIQUE (norm, abschnitt)
```

`(norm, abschnitt)` ist der natürliche Schlüssel: eine Zeile pro Norm + Abschnitt,
damit Re-Seed und manuelle Bearbeitung per UPSERT statt Duplikat laufen.

## Setup

### 1. Lakebase-Instanz + native Passwort-Rolle anlegen
Wie im [Referenz-Repo](https://github.com/EcZachly/databricks-lakebase-app-day-1):
Lakebase-Instanz erstellen, Passwort-Authentifizierung aktivieren, Rolle mit
Passwort anlegen, Connection-URL kopieren:
```
postgresql://<role>:<password>@<host>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require
```

### 2. Secret speichern (einmalig, aus einem Databricks-Notebook)
```python
%sh python setup_secrets.py    # fragt via getpass nach der Lakebase-URL
```

### 3. Lokal entwickeln
```bash
cp .env.example .env      # LAKEBASE_URL eintragen (nicht committen)
pip install -r requirements.txt
python app.py             # http://127.0.0.1:8000
curl -X POST localhost:8000/api/seed   # data.csv laden
```

### 4. Als Databricks App deployen (ohne CLI)
Git-Folder im Workspace anlegen → **Compute > Apps > Create app** → als Quelle den
Ordner `baunorm-lakebase-app/` (mit `app.py` + `app.yaml`) wählen → **Deploy**.
`app.yaml` liefert Scope/Key für die Lakebase-URL. Danach `/healthz` prüfen und
`/api/seed` einmal aufrufen.

## Change Data Feed (CDF) aktivieren

Damit Lakebase-Änderungen automatisch als Delta-Tabellen in Unity Catalog landen
(kein Debezium, kein Connector):

### 1. `REPLICA IDENTITY FULL` – schon erledigt
`lakebase.ensure_table_with_cdf()` setzt das beim Anlegen der `normen`-Tabelle
automatisch. Manuell zur Kontrolle:
```sql
ALTER TABLE normen REPLICA IDENTITY FULL;
SELECT * FROM wal2delta.tables;   -- prüfen, welche Tabellen qualifizieren
```
> Tabellen mit dem Setting aber 0 Zeilen werden erst nach dem ersten Insert
> (z. B. nach `/api/seed`) aufgenommen.

### 2. CDF in der Lakebase-UI starten
**Lakebase > Lakebase CDF > Start** → Datenbank `databricks_postgres`, Schema
`public` → Ziel-Katalog/-Schema in Unity Catalog wählen → bestätigen.

Jede Tabelle bekommt dann `lb_<table>_history` (z. B. `lb_normen_history`), ~alle
15 s aktualisiert, mit Metadaten-Spalten (`_pg_change_type`, `_pg_lsn`, `_pg_xid`,
`_timestamp`, `_sort_by`) für nachgelagerte DLT-Pipelines (Silber/Gold).

## Drei Architekturen für dasselbe Problem

„Deutsche Bau-Normen durchsuchbar machen" – drei Wege auf Databricks, je nach
Ziel:

| | **A) Vector Search + Streamlit** | **B) Lakebase + Flask + CDF** *(dieses Repo)* | **C) Delta + Databricks SQL** |
|---|---|---|---|
| **Kernidee** | Normtexte embedden, semantische Ähnlichkeitssuche | OLTP-Postgres als App-Backend, CDF spiegelt nach Lakehouse | Batch-Tabellen im Lakehouse, SQL-Warehouse-Abfragen |
| **Suche** | Semantisch („feuerhemmend" ≈ „Brandschutz 30 Min") | Exakte/Substring-Suche (ILIKE), deterministisch | SQL/BI, kein freies Suchfeld |
| **Schreiben/Pflege** | Re-Index nötig, kein Live-Editing | Direkt über App, sofort sichtbar | Batch-Ingest (Job/Notebook) |
| **Latenz** | Mittel (Embedding + Vektorsuche) | Sehr niedrig (Point-Lookup in Postgres) | Höher (Warehouse-Query, analytisch) |
| **Analyse-Layer** | Separat | **Automatisch via CDF → Delta** | Nativ (ist schon Delta) |
| **Stärke** | Findet fachlich verwandte Begriffe | Interaktive App + Lakehouse in einem | Große Aggregationen, BI-Dashboards |
| **Schwäche** | Overkill bei exakten Normnummern; „erfindet" evtl. Nähe | Keine Semantik out-of-the-box | Nicht für Live-App / Einzelabfrage gedacht |
| **Passt, wenn** | Nutzer beschreiben Probleme unscharf | Team pflegt geprüfte Werte + will BI obendrauf | Reines Reporting über viele Normen |

**Warum hier B:** baunorm-finder lebt von **exakten, geprüften Werten** (Normnummer +
Abschnitt + Zahl). Semantische Suche (A) kann Nähe „erfinden" – gefährlich bei
sicherheitsrelevanten Bemessungswerten. Ein reines Warehouse (C) taugt nicht als
interaktive Nachschlage-App. Lakebase (B) gibt Tragwerksplanern eine schnelle,
deterministische App **und** – über CDF – automatisch eine Analyse-Schicht im
Lakehouse, ohne zweite Pipeline.

## Hinweise

- Lakebase-Auth: einzelnes `LAKEBASE_URL`-Secret auf eine native Postgres-Rolle mit
  statischem Passwort – keine Token-Refresh-Logik.
- Suche ist Substring-`ILIKE` (wie das Original). Für viele Einträge später auf
  Postgres-Volltextsuche (`to_tsvector('german', …)`) oder ein `pg_trgm`-GIN-Index
  umstellen.
- `data.csv`-Datenstand (teils verifiziert, teils Platzhalter) siehe
  [Haupt-README](../README.md).
