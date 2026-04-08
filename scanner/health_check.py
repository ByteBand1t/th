#!/usr/bin/env python3
"""OllamaScout Health Check — probes known models, updates availability scores."""
from __future__ import annotations

import json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path("/app/.env"))

import httpx, yaml

DATA_DIR     = Path("/data")
RESULTS_FILE = DATA_DIR / "results.json"
HEALTH_FILE  = DATA_DIR / "health.json"
CONFIG_FILE  = Path("/app/config.yaml")

PROBE_TIMEOUT = 5.0
BENCH_TIMEOUT = 45.0


def load_config():
    raw = CONFIG_FILE.read_text()
    def _expand(m):
        expr = m.group(1)
        if ":-" in expr:
            var, default = expr.split(":-", 1)
            return os.environ.get(var, default)
        return os.environ.get(expr, m.group(0))
    return yaml.safe_load(re.sub(r"\$\{([^}]+)\}", _expand, raw))


def load_results():
    if RESULTS_FILE.exists():
        try: return json.loads(RESULTS_FILE.read_text())
        except: pass
    return {"candidates": []}


def save_results(data):
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _quick_bench(ip, port, model):
    payload = {
        "model": model, "prompt": "Say only: OK",
        "stream": True, "options": {"num_predict": 10},
    }
    t0 = time.perf_counter()
    ttft, tokens, text = None, 0, ""
    with httpx.stream("POST", f"http://{ip}:{port}/api/generate",
                      json=payload, timeout=BENCH_TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line: continue
            try: chunk = json.loads(line)
            except: continue
            frag = chunk.get("response", "")
            if frag:
                if ttft is None: ttft = time.perf_counter() - t0
                tokens += 1; text += frag
            if chunk.get("done"): break
    elapsed = time.perf_counter() - t0
    return (ttft or elapsed), (tokens / elapsed if elapsed else 0), text


def _remove_from_litellm(base_url, master_key, model_name, log_fn):
    headers = {"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"}
    try:
        r = httpx.get(f"{base_url}/model/info", headers=headers, timeout=10)
        r.raise_for_status()
        for m in r.json().get("data", []):
            if m.get("model_name") == model_name:
                mid = m.get("model_info", {}).get("id", "")
                if mid:
                    httpx.post(f"{base_url}/model/delete", headers=headers,
                               json={"id": mid}, timeout=10).raise_for_status()
                    log_fn(f"  Removed from LiteLLM: {model_name}")
                    return True
    except Exception as e:
        log_fn(f"  Error removing {model_name}: {e}")
    return False


def _update_availability(c: dict, alive: bool):
    """Update running availability score on the candidate dict."""
    checks = c.get("availabilityChecks", 0) + 1
    alive_count = c.get("availabilityAlive", 0) + (1 if alive else 0)
    c["availabilityChecks"] = checks
    c["availabilityAlive"]  = alive_count
    c["availability"]       = round(alive_count / checks, 3)


def run_health_check(log_fn=print) -> dict:
    cfg      = load_config()
    lc       = cfg["litellm"]
    bc       = cfg.get("benchmark", {})
    max_ttft = float(bc.get("max_ttft_seconds", 15.0)) * 1.5   # warn threshold
    min_tps  = float(bc.get("min_tokens_per_second", 3.0)) * 0.6
    # Dead threshold: 2.5x worse than warn
    dead_ttft = max_ttft * 2.5
    dead_tps  = min_tps  * 0.4

    data       = load_results()
    candidates = data.get("candidates", [])
    to_check   = [c for c in candidates if c.get("status") in ("added", "slow")]
    now        = datetime.now(timezone.utc).isoformat()

    log_fn(f"Health check — {len(to_check)} models to check")
    report_results = []

    for c in to_check:
        ip    = c.get("ip")
        port  = c.get("port", 11434)
        model = c.get("model")
        name  = c.get("litellmName") or ""

        log_fn(f"\n  [{model}] @ {ip}:{port}")

        # Stage 1: reachable + model still available
        try:
            r    = httpx.get(f"http://{ip}:{port}/api/tags",
                             timeout=httpx.Timeout(PROBE_TIMEOUT, connect=2.0))
            tags = r.json().get("models", []) if r.status_code == 200 else []
            names = [m.get("name", "") for m in tags]
            if not any(model == n or model.split(":")[0] in n for n in names):
                raise ValueError(f"Model '{model}' not in /api/tags")
        except Exception as e:
            log_fn(f"    ✗ Unreachable: {e}")
            if name: _remove_from_litellm(lc["base_url"], lc["master_key"], name, log_fn)
            c.update({"status": "dead", "deadReason": str(e)})
            _update_availability(c, False)
            report_results.append({"model": model, "ip": ip, "status": "dead", "reason": str(e)})
            continue

        # Stage 2: quick benchmark
        try:
            ttft, tps, _ = _quick_bench(ip, port, model)
            log_fn(f"    TTFT={ttft:.2f}s  TPS={tps:.1f}")

            if ttft > dead_ttft or tps < dead_tps:
                reason = f"TTFT {ttft:.1f}s>{dead_ttft:.0f}s" if ttft > dead_ttft else f"TPS {tps:.1f}<{dead_tps:.1f}"
                log_fn(f"    ✗ Too slow → removing: {reason}")
                if name: _remove_from_litellm(lc["base_url"], lc["master_key"], name, log_fn)
                c.update({"status": "dead", "deadReason": reason,
                          "ttft": round(ttft,3), "tps": round(tps,1)})
                _update_availability(c, False)
                report_results.append({"model": model, "ip": ip, "status": "dead",
                                       "reason": reason, "ttft": ttft, "tps": tps})

            elif ttft > max_ttft or tps < min_tps:
                reason = f"TTFT {ttft:.1f}s" if ttft > max_ttft else f"TPS {tps:.1f}"
                log_fn(f"    ⚠ Slow but keeping: {reason}")
                c.update({"status": "slow", "slowReason": reason,
                          "ttft": round(ttft,3), "tps": round(tps,1), "lastHealthCheck": now})
                _update_availability(c, True)
                report_results.append({"model": model, "ip": ip, "status": "slow",
                                       "reason": reason, "ttft": ttft, "tps": tps})
            else:
                log_fn(f"    ✓ Healthy | avail={c.get('availability', 1.0):.0%}")
                c.update({"status": "added", "deadReason": None, "slowReason": None,
                          "ttft": round(ttft,3), "tps": round(tps,1), "lastHealthCheck": now})
                _update_availability(c, True)
                report_results.append({"model": model, "ip": ip, "status": "alive",
                                       "ttft": ttft, "tps": tps})

        except Exception as e:
            log_fn(f"    ✗ Benchmark failed: {e}")
            if name: _remove_from_litellm(lc["base_url"], lc["master_key"], name, log_fn)
            c.update({"status": "dead", "deadReason": str(e)})
            _update_availability(c, False)
            report_results.append({"model": model, "ip": ip, "status": "dead", "reason": str(e)})

    save_results(data)

    summary = {
        "lastCheck": now,
        "checked":   len(to_check),
        "alive":     sum(1 for r in report_results if r["status"] == "alive"),
        "slow":      sum(1 for r in report_results if r["status"] == "slow"),
        "dead":      sum(1 for r in report_results if r["status"] == "dead"),
        "results":   report_results,
    }
    HEALTH_FILE.write_text(json.dumps(summary, indent=2))
    log_fn(f"\nDone: {summary['alive']} alive · {summary['slow']} slow · {summary['dead']} dead/removed")
    return summary


if __name__ == "__main__":
    run_health_check()
