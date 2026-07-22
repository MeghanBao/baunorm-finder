# Normen-Nachschlage-Tool - Backend
# Liest data.csv beim Start ein und stellt eine Such-API bereit.
# Start: python server.py  dann Browser: http://127.0.0.1:8010

import csv
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data.csv"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Normen-Nachschlage-Tool")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def load_rows():
    with open(DATA_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


ROWS = load_rows()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/search")
def search(q: str = Query("", description="Suchbegriff")):
    q = q.strip().lower()
    if not q:
        return {"total": len(ROWS), "results": []}

    fields = ["Norm", "Abschnitt", "Stichworte", "Zusammenfassung"]
    matched = [r for r in ROWS if any(q in (r.get(f) or "").lower() for f in fields)]

    return {
        "total": len(matched),
        "results": [
            {
                "code": r.get("Norm", ""),
                "article": r.get("Abschnitt", ""),
                "keywords": r.get("Stichworte", ""),
                "summary": r.get("Zusammenfassung", ""),
                "value": r.get("Wert", ""),
                "note": r.get("Anmerkung", ""),
            }
            for r in matched
        ],
    }


@app.get("/api/reload")
def reload_data():
    """Neu einlesen der data.csv, ohne den Server neu zu starten."""
    global ROWS
    ROWS = load_rows()
    return {"status": "ok", "count": len(ROWS)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
