"""OllamaScout — backend.py"""
from __future__ import annotations

import json, os, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

DATA_DIR      = Path("/data")
RESULTS_FILE  = DATA_DIR / "results.json"
HEALTH_FILE   = DATA_DIR / "health.json"
LOCK_FILE     = DATA_DIR / ".scan_running"
HLOCK_FILE    = DATA_DIR / ".health_running"
LOG_FILE      = DATA_DIR / "scan.log"
HLOG_FILE     = DATA_DIR / "health.log"
SCHEDULE_FILE = DATA_DIR / "schedule.json"
USER_MODELS   = DATA_DIR / "user_models.json"
SCANNER       = Path("/app/scanner/main.py")
HEALTH_CHK    = Path("/app/scanner/health_check.py")

app = FastAPI(title="OllamaScout", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR.mkdir(exist_ok=True)

DEFAULT_SCHEDULE = {
    "full_scan_enabled":           True,
    "full_scan_weekday":           6,
    "full_scan_hour":              3,
    "health_check_enabled":        True,
    "health_check_interval_hours": 12,
    "next_health_check":           None,
    "next_full_scan":              None,
}


# ── Helpers ───────────────────────────────────────────────────

def load_schedule() -> dict:
    if SCHEDULE_FILE.exists():
        try: return {**DEFAULT_SCHEDULE, **json.loads(SCHEDULE_FILE.read_text())}
        except: pass
    return DEFAULT_SCHEDULE.copy()


def save_schedule(s: dict):
    SCHEDULE_FILE.write_text(json.dumps(s, indent=2))


def load_results() -> dict:
    if RESULTS_FILE.exists():
        try: return json.loads(RESULTS_FILE.read_text())
        except: pass
    return {"lastScan": None, "status": "never_run", "hostsScanned": 0, "candidates": [], "log": []}


def load_health() -> dict:
    if HEALTH_FILE.exists():
        try: return json.loads(HEALTH_FILE.read_text())
        except: pass
    return {"lastCheck": None, "checked": 0, "alive": 0, "slow": 0, "dead": 0, "results": []}


def load_user_models() -> list:
    if USER_MODELS.exists():
        try: return json.loads(USER_MODELS.read_text())
        except: pass
    return []


def save_user_models(models: list):
    USER_MODELS.write_text(json.dumps(models, indent=2))


# ── Background runners ────────────────────────────────────────

def _run_scan():
    LOCK_FILE.touch()
    try:
        result = subprocess.run(
            [sys.executable, str(SCANNER)],
            capture_output=True, text=True, cwd="/app/scanner",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        LOG_FILE.write_text(result.stdout + "\n" + result.stderr)
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def _run_health_check():
    HLOCK_FILE.touch()
    try:
        result = subprocess.run(
            [sys.executable, str(HEALTH_CHK)],
            capture_output=True, text=True, cwd="/app/scanner",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        HLOG_FILE.write_text(result.stdout + "\n" + result.stderr)
    finally:
        HLOCK_FILE.unlink(missing_ok=True)


# ── Scheduler ─────────────────────────────────────────────────

def _scheduler():
    while True:
        try:
            from datetime import timedelta
            s   = load_schedule()
            now = datetime.now(timezone.utc)

            if s.get("health_check_enabled") and not HLOCK_FILE.exists() and not LOCK_FILE.exists():
                nxt = s.get("next_health_check")
                if nxt is None or datetime.fromisoformat(nxt) <= now:
                    threading.Thread(target=_run_health_check, daemon=True).start()
                    s["next_health_check"] = (now + timedelta(hours=s["health_check_interval_hours"])).isoformat()
                    save_schedule(s)

            if s.get("full_scan_enabled") and not LOCK_FILE.exists():
                nxt = s.get("next_full_scan")
                if nxt is None:
                    target_wd = s["full_scan_weekday"]
                    target_hr = s["full_scan_hour"]
                    days_ahead = (target_wd - now.weekday()) % 7
                    if days_ahead == 0 and now.hour >= target_hr:
                        days_ahead = 7
                    run_at = (now + timedelta(days=days_ahead)).replace(
                        hour=target_hr, minute=0, second=0, microsecond=0)
                    s["next_full_scan"] = run_at.isoformat()
                    save_schedule(s)
                elif datetime.fromisoformat(nxt) <= now:
                    threading.Thread(target=_run_scan, daemon=True).start()
                    s["next_full_scan"] = (now + timedelta(days=7)).isoformat()
                    save_schedule(s)
        except Exception:
            pass
        time.sleep(60)


threading.Thread(target=_scheduler, daemon=True).start()


# ── Scan API ──────────────────────────────────────────────────

@app.get("/api/results")
def get_results():
    data = load_results()
    data["scanning"]       = LOCK_FILE.exists()
    data["healthChecking"] = HLOCK_FILE.exists()
    data["health"]         = load_health()
    data["schedule"]       = load_schedule()
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


@app.post("/api/health-check")
def trigger_health(background_tasks: BackgroundTasks):
    if HLOCK_FILE.exists():
        raise HTTPException(status_code=409, detail="Health check already running")
    background_tasks.add_task(_run_health_check)
    return {"message": "Health check started"}


@app.get("/api/log")
def get_log():
    return {"log": LOG_FILE.read_text() if LOG_FILE.exists() else ""}


@app.get("/api/health-log")
def get_health_log():
    return {"log": HLOG_FILE.read_text() if HLOG_FILE.exists() else ""}


# ── Schedule API ──────────────────────────────────────────────

class ScheduleUpdate(BaseModel):
    full_scan_enabled:           bool = True
    full_scan_weekday:           int  = 6
    full_scan_hour:              int  = 3
    health_check_enabled:        bool = True
    health_check_interval_hours: int  = 12


@app.get("/api/schedule")
def get_schedule():
    return load_schedule()


@app.post("/api/schedule")
def update_schedule(s: ScheduleUpdate):
    current = load_schedule()
    current.update(s.dict())
    current["next_full_scan"]    = None
    current["next_health_check"] = None
    save_schedule(current)
    return current


# ── Target Models API ─────────────────────────────────────────

class ModelEntry(BaseModel):
    name:       str
    min_size_b: float = 0.0


@app.get("/api/models/targets")
def get_target_models():
    import yaml, re
    try:
        raw = Path("/app/config.yaml").read_text()
        def _expand(m):
            expr = m.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.environ.get(var, default)
            return os.environ.get(expr, m.group(0))
        cfg = yaml.safe_load(re.sub(r"\$\{([^}]+)\}", _expand, raw))
        base = cfg.get("target_models", [])
        base_norm = [{"name": t, "min_size_b": 0, "source": "config"} if isinstance(t, str)
                     else {**t, "source": "config"} for t in base]
    except Exception:
        base_norm = []
    user = [{"name": m["name"], "min_size_b": m.get("min_size_b", 0), "source": "user"}
            for m in load_user_models()]
    merged = {m["name"]: m for m in base_norm}
    for m in user:
        merged[m["name"]] = m
    return {"models": list(merged.values())}


@app.post("/api/models/targets")
def add_target_model(entry: ModelEntry):
    models = load_user_models()
    if any(m["name"] == entry.name for m in models):
        raise HTTPException(status_code=409, detail=f"{entry.name} already in user list")
    models.append({"name": entry.name, "min_size_b": entry.min_size_b})
    save_user_models(models)
    return {"message": f"Added {entry.name}"}


@app.delete("/api/models/targets/{name}")
def remove_target_model(name: str):
    models = load_user_models()
    before = len(models)
    models = [m for m in models if m["name"] != name]
    if len(models) == before:
        raise HTTPException(status_code=404, detail=f"{name} not in user model list")
    save_user_models(models)
    return {"message": f"Removed {name}"}


# ── Probe / Benchmark / Manual Add ───────────────────────────

class ProbeRequest(BaseModel):
    ip:   str
    port: int = 11434


class BenchmarkRequest(BaseModel):
    ip:     str
    port:   int
    models: List[str]


class AddRequest(BaseModel):
    ip:       str
    port:     int
    org:      str   = "manual"
    country:  str   = "??"
    model:    str
    ttft:     float
    tps:      float
    response: str


@app.post("/api/probe")
def probe_host(req: ProbeRequest):
    try:
        r = httpx.get(f"http://{req.ip}:{req.port}/api/tags", timeout=6.0)
        r.raise_for_status()
        return {
            "ip": req.ip, "port": req.port, "reachable": True,
            "models": [{"name": m.get("name", ""), "size": m.get("size", 0)}
                       for m in r.json().get("models", [])],
        }
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Cannot connect to {req.ip}:{req.port}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/benchmark-single")
def benchmark_single(req: BenchmarkRequest):
    import time as t
    base_url = f"http://{req.ip}:{req.port}"
    results  = []
    for model in req.models:
        ttfts, tpss, last_text, error = [], [], "", None
        for _ in range(3):
            try:
                t0 = t.perf_counter(); ttft = None; tokens = 0; text = ""
                with httpx.stream(
                    "POST", f"{base_url}/api/generate",
                    json={"model": model, "prompt": "Hallo, wie geht es dir?",
                          "stream": True, "options": {"num_predict": 80}},
                    timeout=90.0,
                ) as resp:
                    if resp.status_code == 401:
                        raise PermissionError("401 Unauthorized")
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line: continue
                        try: chunk = json.loads(line)
                        except: continue
                        frag = chunk.get("response", "")
                        if frag:
                            if ttft is None: ttft = t.perf_counter() - t0
                            tokens += 1; text += frag
                        if chunk.get("done"): break
                elapsed = t.perf_counter() - t0
                ttfts.append(ttft or elapsed)
                tpss.append(tokens / elapsed if elapsed else 0)
                last_text = text
            except Exception as e:
                error = str(e); break
        if error or not ttfts:
            results.append({"model": model, "ok": False, "error": error or "No data",
                            "ttft": None, "tps": None, "response": ""})
        else:
            avg_ttft = sum(ttfts) / len(ttfts)
            avg_tps  = sum(tpss)  / len(tpss)
            results.append({
                "model": model,
                "ok":    avg_ttft <= 15.0 and avg_tps >= 3.0,
                "ttft":  round(avg_ttft, 3),
                "tps":   round(avg_tps, 1),
                "response": last_text,
                "error": None,
            })
    return {"ip": req.ip, "port": req.port, "results": results}


def _do_add(req: AddRequest):
    import re
    sys.path.insert(0, "/app/scanner")
    os.chdir("/app/scanner")
    from litellm_manager import LiteLLMManager
    from shodan_scanner import OllamaHost

    log_lines: list[str] = []
    mgr  = LiteLLMManager(
        os.environ.get("LITELLM_BASE_URL", ""),
        os.environ.get("LITELLM_MASTER_KEY", ""),
        log_fn=log_lines.append,
    )
    host = OllamaHost(ip=req.ip, port=req.port, country=req.country, org=req.org)

    class FC:
        model_name        = req.model
        matched_target    = req.model.split(":")[0]
        ttft              = req.ttft
        tokens_per_second = req.tps
        response_text     = req.response
        benchmark_ok      = True
        benchmark_error   = None

        @property
        def litellm_model_string(self):
            return f"ollama/{req.model}"

        @property
        def litellm_model_name(self):
            return f"{req.model.replace(':', '-').replace('/', '-')}@{req.ip.replace('.', '-')}"

        @property
        def pool_name(self):
            base = self.matched_target
            m    = re.search(r"(\d+)b", req.model.lower())
            return f"{base}-{m.group(1)}b-pool" if m else f"{base}-pool"

    c = FC()
    c.__class__.host = host  # fix: assign host after class definition

    mgr.add_model(c)

    data  = load_results()
    entry = {
        "id":          len(data.get("candidates", [])) + 1,
        "ip":          req.ip,
        "port":        req.port,
        "country":     req.country,
        "org":         req.org,
        "model":       req.model,
        "matched":     c.matched_target,
        "ttft":        req.ttft,
        "tps":         req.tps,
        "response":    req.response,
        "status":      "added",
        "failReason":  None,
        "litellmName": c.litellm_model_name,
        "manual":      True,
    }
    data.setdefault("candidates", []).append(entry)
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"message": f"Added {req.model}", "log": log_lines, "entry": entry}


@app.post("/api/add-manual")
def add_manual(req: AddRequest):
    return _do_add(req)


@app.post("/api/add-anyway")
def add_anyway(req: AddRequest):
    return _do_add(req)


# ── Chat ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    ip:      str
    port:    int
    model:   str
    message: str
    history: list = []


@app.post("/api/chat")
def chat_with_model(req: ChatRequest):
    import time as t
    messages = req.history + [{"role": "user", "content": req.message}]
    t0 = t.perf_counter()
    try:
        r = httpx.post(
            f"http://{req.ip}:{req.port}/api/chat",
            json={"model": req.model, "messages": messages, "stream": False},
            timeout=60.0,
        )
        r.raise_for_status()
        return {
            "response": r.json().get("message", {}).get("content", ""),
            "elapsed":  round(t.perf_counter() - t0, 2),
        }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.delete("/api/model/{model_name}")
def delete_model(model_name: str):
    data = load_results()
    data["candidates"] = [c for c in data["candidates"] if c.get("litellmName") != model_name]
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"message": f"Removed {model_name}"}


# ── Frontend ──────────────────────────────────────────────────

FRONTEND = Path("/app/frontend")


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND / "index.html"))


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
