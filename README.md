# Normen-Nachschlage-Tool (Deutschland)

Stichwort eingeben (z. B. "Mindestbewehrung"), sofort die passende Norm, Abschnittsnummer
und den relevanten Wert finden — ohne PDFs zu durchsuchen.

> **🔎 Live-Demo → https://meghanbao.github.io/baunorm-finder/demo.html**
> Einfach anklicken — läuft direkt im Browser, keine Installation.

## Demo ohne Installation

Zwei Wege, beide ohne Server und ohne Installation:

- **Online:** [meghanbao.github.io/baunorm-finder/demo.html](https://meghanbao.github.io/baunorm-finder/demo.html)
  öffnen (via GitHub Pages gehostet).
- **Offline:** `demo.html` doppelt anklicken — die Demo ist self-contained
  (Beispieldaten eingebettet) und läuft komplett offline, auch per `file://`.

Enthält einen Teil bereits verifizierter Daten (Brandschutzklassen), Rest noch
Platzhalter (siehe Hinweisbanner in der Demo). Ausprobieren: "feuerhemmend",
"nichtbrennbar", "Erdbeben".

## Produktivbetrieb starten

```bash
cd baunorm-finder
pip install -r requirements.txt
python server.py
```

Browser öffnen: **http://127.0.0.1:8010**

Der Produktivbetrieb liest `data.csv` ein und stellt eine Such-API bereit — hier
werden später die echten, geprüften Normwerte gepflegt.

## Datenstand: teils verifiziert, teils noch Platzhalter

`data.csv` enthält jetzt 5 Einträge, gespeist aus der **Muster-Verwaltungsvorschrift
Technische Baubestimmungen (MVV TB) 2025/1** — amtlich, kostenfrei vom DIBt
veröffentlicht (354 Seiten, Stand 20. Mai 2025, mit Druckfehlerberichtigungen).

- **✅ Vollständig verifiziert** (2 Einträge): Baustoffklassen nach DIN 4102-1
  (nichtbrennbar/schwerentflammbar/normalentflammbar) und Feuerwiderstandsklassen
  nach DIN 4102-2 (feuerhemmend = 30 Min, hochfeuerhemmend = 60 Min,
  feuerbeständig = 90 Min) — Werte direkt aus dem amtlichen Text übernommen
- **⚠ Ausgabe verifiziert, Zahlenwert noch offen** (2 Einträge): Eurocode 1
  (Einwirkungen) und Eurocode 2 (Beton) — die MVV TB bestätigt, welche exakte
  Norm-/NA-Ausgabe gilt, aber die konkreten Bemessungswerte/Formeln stehen im
  kostenpflichtigen Eurocode-Volltext selbst, nicht in der MVV TB
- **🔧 Wichtige Korrektur**: Erdbebennachweise laufen in Deutschland über die
  **nationale Norm DIN 4149:2005-04**, nicht über Eurocode 8 — das war eine
  falsche Annahme in einer früheren Version dieses Tools

Für die noch offenen Zahlenwerte (Eurocode 1/2 Formeln, weitere Themen) gilt
weiterhin: keine Werte erfunden, echter Zugriff auf den Normtext nötig
(Firmenabo, Bibliothek oder Kauf bei DIN Media/Beuth Verlag).

### Besonders zu beachten

- Eurocodes sind EU-weit einheitlich, aber jedes Land hat einen eigenen
  **Nationalen Anhang (National Annex, NA)** — erst der deutsche DIN EN xxxx/NA
  enthält die tatsächlich gültigen Werte, nicht der Eurocode-Basistext allein
- Brandschutz, Rettungswege usw. unterliegen zusätzlich den
  **Landesbauordnungen (LBO)** der einzelnen Bundesländer, die strenger sein können
  als DIN/Eurocode — diese müssen zusätzlich geprüft werden
- Erdbebenzonen sowie Schneelast-/Windlastzonen haben eigene, länderspezifische
  Karten — keine Werte aus anderen Ländern übernehmen

## Echte Daten eintragen

`data.csv` mit Excel/WPS öffnen, Spalten wie vorhanden befüllen, als CSV speichern
(UTF-8 beibehalten):

| Spalte | Bedeutung | Beispiel |
|---|---|---|
| Norm | Vollständige Bezeichnung + Ausgabejahr | DIN EN 1992-1-1/NA:2013-04 |
| Abschnitt | Genaue Abschnittsnummer | 9.2.1.1 |
| Stichworte | Mit Semikolon getrennt, für die Suche | Mindestbewehrung;Längsbewehrung |
| Zusammenfassung | Kurzbeschreibung des Inhalts | Mindestbewehrung für Längsbewehrung bei Balken… |
| Wert | Konkrete Zahl/Formel/Tabelle | 0,26·fctm/fyk·bt·d, mind. 0,0013·bt·d (Beispiel, Original prüfen) |
| Anmerkung | Randbedingungen, Stolperfallen | Mit Werten im Nationalen Anhang abgleichen |

Nach dem Speichern reicht `GET /api/reload`, um die Daten neu einzulesen —
kein Neustart des Servers nötig.

## Quelle

MVV TB 2025/1: https://www.dibt.de/fileadmin/dibt-website/Dokumente/Referat/P5/Technische_Bestimmungen/MVVTB_2025-1.pdf
(amtlich, kostenfrei, DIBt). Enthält u. a. die vollständige Liste aller in
Deutschland bauordnungsrechtlich eingeführten Technischen Baubestimmungen mit
exakten Ausgabedaten der Normen und Nationalen Anhänge — aber nicht die
Normtexte selbst.

## Architektur

- `demo.html` — Standalone-Demo mit eingebetteten Beispieldaten, kein Server nötig;
  online via GitHub Pages unter
  [meghanbao.github.io/baunorm-finder/demo.html](https://meghanbao.github.io/baunorm-finder/demo.html)
- `server.py` — FastAPI-Backend für den Produktivbetrieb, liest `data.csv` ein,
  stellt `/api/search?q=` bereit
- `static/index.html` — Frontend des Produktivbetriebs, ruft die Such-API auf
- Skaliert bei größeren Datenmengen (mehrere hundert Einträge+); kann später auf
  eine echte Volltextsuche (z. B. SQLite FTS5) umgestellt werden

## Databricks-Varianten

Über die schlanke Stichwortsuche hinaus gibt es zwei Databricks-Ausbaustufen:

- **[`baunorm-lakebase-app/`](baunorm-lakebase-app/README.md)** — dieselbe Suche als
  **Databricks App** auf **Lakebase** (managed Postgres) mit Change Data Feed.
- **[`databricks-rag-platform/`](databricks-rag-platform/README.md)** — ein
  **unternehmenstauglicher RAG-Assistent**, vollständig nativ auf Databricks:
  provenance-getaggter Korpus → **Vector Search** → **Mosaic AI Agent**
  (Llama 3.3 70B) → **MLflow-Evaluation** → App-UI mit Zitaten und
  Herkunfts-Badges. Generation ohne externen LLM; deploybar als **ein Asset Bundle**
  (dev/staging/prod).

## Ähnliche Projekte in Deutschland

Es gibt bereits kommerzielle Lösungen mit ähnlicher Stoßrichtung, aber kein
kostenloses, schlankes Tool speziell für die schnelle Stichwortsuche von
Tragwerksplanern:

- **Technische Baubestimmungen / bauvorschriften.de** — über 2.000 DIN-Normen und
  rund 350 Richtlinien, inkl. Eurocodes und Nationalen Anhängen, Volltextsuche,
  aber kostenpflichtiges Abonnement
- **Eurocode-online.de** — offizielle digitale Eurocode-Pakete von DIN, ebenfalls
  kostenpflichtig
- **DIN Bauportal** — offizielles DIN-Portal, eher zum Durchsuchen von
  Normenkatalogen als für schnelle Wertabfragen
- **Bauprofessor** — kostenlose Suchmaschine, aber Fokus auf Baukosten, VOB,
  Fachbegriffe statt Tragwerksnormen
- **"Handwerk & Recht" / Gessner Baurecht (Apps)** — Baurecht (Verträge, Abnahme,
  Gewährleistung), nicht Tragwerksplanung
- **NORM2GO** — prüft nur, ob eine DIN-Norm noch gültig ist (Barcode-Scan)
- **juris Baurecht** — professionelle Rechtsdatenbank, kostenpflichtig, Zielgruppe
  Juristen

Fazit: Eine kostenlose, schlanke Lösung speziell für Tragwerksplaner mit
Stichwort-zu-Wert-Suche scheint eine echte Lücke zu sein.
