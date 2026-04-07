"""
OllamaScout — backend.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

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


# ── Scan API ─────────────────────────────────────────────────

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


# ── Manual Add API ───────────────────────────────────────────

class ProbeRequest(BaseModel):
    ip: str
    port: int = 11434


class BenchmarkRequest(BaseModel):
    ip: str
    port: int
    models: List[str]


class AddRequest(BaseModel):
    ip: str
    port: int
    org: str = "manual"
    country: str = "??"
    model: str
    ttft: float
    tps: float
    response: str


@app.post("/api/probe")
def probe_host(req: ProbeRequest):
    """Check what models are available on a given IP:port."""
    base_url = f"http://{req.ip}:{req.port}"
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=6.0)
        r.raise_for_status()
        models = r.json().get("models", [])
        return {
            "ip": req.ip,
            "port": req.port,
            "reachable": True,
            "models": [
                {
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in models
            ],
        }
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Cannot connect to {req.ip}:{req.port}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Timeout connecting to {req.ip}:{req.port}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/benchmark-single")
def benchmark_single(req: BenchmarkRequest):
    """Benchmark selected models on a given host and return results."""
    import time

    base_url = f"http://{req.ip}:{req.port}"
    results  = []

    for model in req.models:
        ttfts, tpss, last_text = [], [], ""
        error = None

        for run in range(3):
            try:
                payload = {
                    "model": model,
                    "prompt": "Hallo, wie geht es dir?",
                    "stream": True,
                    "options": {"num_predict": 80},
                }
                t0    = time.perf_counter()
                ttft  = None
                tokens = 0
                text   = ""

                with httpx.stream("POST", f"{base_url}/api/generate",
                                  json=payload, timeout=90.0) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            continue
                        frag = chunk.get("response", "")
                        if frag:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            tokens += 1
                            text   += frag
                        if chunk.get("done"):
                            break

                elapsed = time.perf_counter() - t0
                ttfts.append(ttft or elapsed)
                tpss.append(tokens / elapsed if elapsed else 0)
                last_text = text

            except Exception as e:
                error = str(e)
                break

        if error or not ttfts:
            results.append({
                "model": model,
                "ok": False,
                "error": error or "No data",
                "ttft": None,
                "tps": None,
                "response": "",
            })
        else:
            avg_ttft = sum(ttfts) / len(ttfts)
            avg_tps  = sum(tpss)  / len(tpss)
            results.append({
                "model":    model,
                "ok":       avg_ttft <= 5.0 and avg_tps >= 8.0,
                "ttft":     round(avg_ttft, 3),
                "tps":      round(avg_tps, 1),
                "response": last_text,
                "error":    None,
            })

    return {"ip": req.ip, "port": req.port, "results": results}


@app.post("/api/add-manual")
def add_manual(req: AddRequest):
    """Manually add a benchmarked model to LiteLLM and results.json."""
    import re
    import sys
    sys.path.insert(0, "/app/scanner")
    from litellm_manager import LiteLLMManager
    from ollama_checker import ModelCandidate
    from shodan_scanner import OllamaHost

    litellm_base = os.environ.get("LITELLM_BASE_URL", "")
    litellm_key  = os.environ.get("LITELLM_MASTER_KEY", "")

    log_lines: list[str] = []
    mgr = LiteLLMManager(litellm_base, litellm_key, log_fn=log_lines.append)

    # Build a fake candidate
    host = OllamaHost(ip=req.ip, port=req.port, country=req.country, org=req.org)
    # Find matched target
    matched = req.model.split(":")[0]

    class FakeCandidate:
        model_name          = req.model
        matched_target      = matched
        ttft                = req.ttft
        tokens_per_second   = req.tps
        response_text       = req.response
        benchmark_ok        = True
        benchmark_error     = None

        @property
        def litellm_model_string(self): return f"ollama/{req.model}"
        @property
        def litellm_model_name(self):
            safe = req.model.replace(":", "-").replace("/", "-")
            return f"{safe}@{req.ip.replace('.', '-')}"
        @property
        def pool_name(self):
            base = matched.replace(":", "-")
            m = re.search(r"(\d+(?:\.\d+)?)b", req.model.lower())
            size = int(float(m.group(1))) if m else 0
            return f"{base}-{size}b-pool" if size else f"{base}-pool"
        host = host

    c = FakeCandidate()
    mgr.add_model(c)

    # Also append to results.json
    data = load_results()
    from datetime import datetime, timezone
    new_entry = {
        "id":          len(data.get("candidates", [])) + 1,
        "ip":          req.ip,
        "port":        req.port,
        "country":     req.country,
        "org":         req.org,
        "model":       req.model,
        "matched":     matched,
        "ttft":        req.ttft,
        "tps":         req.tps,
        "response":    req.response,
        "status":      "added",
        "failReason":  None,
        "litellmName": c.litellm_model_name,
        "manual":      True,
    }
    data.setdefault("candidates", []).append(new_entry)
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return {"message": f"Added {req.model}", "log": log_lines, "entry": new_entry}


# ── Frontend ─────────────────────────────────────────────────

FRONTEND = Path("/app/frontend")

@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND / "index.html"))

app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
