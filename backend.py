"""
OllamaScout — backend.py
FastAPI app: serves the dashboard + scan API.
Results are stored in /data/results.json (Docker volume).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

DATA_DIR     = Path("/data")
RESULTS_FILE = DATA_DIR / "results.json"
LOCK_FILE    = DATA_DIR / ".scan_running"
LOG_FILE     = DATA_DIR / "scan.log"
SCANNER      = Path("/app/scanner/main.py")

app = FastAPI(title="OllamaScout", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Helpers ──────────────────────────────────────────────────

def load_results() -> dict:
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    return {"lastScan": None, "status": "never_run", "hostsScanned": 0, "candidates": [], "log": []}


def _run_scan():
    LOCK_FILE.touch()
    try:
        result = subprocess.run(
            [sys.executable, str(SCANNER)],
            capture_output=True, text=True,
            cwd="/app/scanner",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        LOG_FILE.write_text(result.stdout + "\n" + result.stderr)
    finally:
        LOCK_FILE.unlink(missing_ok=True)


# ── API routes ───────────────────────────────────────────────

@app.get("/api/results")
def get_results():
    data = load_results()
    data["scanning"] = LOCK_FILE.exists()
    return data


@app.get("/api/status")
def get_status():
    data = load_results()
    candidates = data.get("candidates", [])
    return {
        "scanning":        LOCK_FILE.exists(),
        "lastScan":        data.get("lastScan"),
        "hostsScanned":    data.get("hostsScanned", 0),
        "totalCandidates": len(candidates),
        "added":           sum(1 for c in candidates if c.get("status") == "added"),
        "failed":          sum(1 for c in candidates if c.get("status") == "failed"),
    }


@app.post("/api/scan")
def trigger_scan(background_tasks: BackgroundTasks):
    if LOCK_FILE.exists():
        raise HTTPException(status_code=409, detail="Scan already running")
    background_tasks.add_task(_run_scan)
    return {"message": "Scan started"}


@app.get("/api/log")
def get_log():
    return {"log": LOG_FILE.read_text() if LOG_FILE.exists() else ""}


@app.delete("/api/model/{model_name}")
def delete_model(model_name: str):
    data = load_results()
    data["candidates"] = [c for c in data["candidates"] if c.get("litellmName") != model_name]
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"message": f"Removed {model_name}"}


# ── Frontend ─────────────────────────────────────────────────

FRONTEND = Path("/app/frontend")

@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND / "index.html"))

app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
