#!/usr/bin/env python3
"""OllamaScout Scanner — writes results to /data/results.json"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path("/app/.env"))

import yaml
from shodan_scanner import discover_hosts
from ollama_checker import discover_candidates, benchmark_candidate
from litellm_manager import LiteLLMManager

DATA_DIR     = Path("/data")
RESULTS_FILE = DATA_DIR / "results.json"
CONFIG_FILE  = Path("/app/config.yaml")
DATA_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    raw = CONFIG_FILE.read_text()

    def _expand(m):
        expr = m.group(1)
        if ":-" in expr:
            var, default = expr.split(":-", 1)
            return os.environ.get(var, default)
        return os.environ.get(expr, m.group(0))

    return yaml.safe_load(re.sub(r"\$\{([^}]+)\}", _expand, raw))


def save(data: dict):
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    cfg = load_config()
    log = []

    def L(msg):
        print(msg, flush=True)
        log.append(msg)

    results = {
        "lastScan":     datetime.now(timezone.utc).isoformat(),
        "status":       "running",
        "hostsScanned": 0,
        "candidates":   [],
        "log":          log,
    }
    save(results)

    # ── Step 1 — Shodan ──────────────────────────────────────
    L("Step 1 — Shodan Discovery")
    sc = cfg["shodan"]
    hosts = discover_hosts(
        api_key=sc["api_key"],
        queries=sc.get("queries", sc.get("query", "ollama port:11434")),
        max_results=sc.get("max_results", 50),
        exclude_orgs=sc.get("exclude_orgs", []),
        log_fn=L,
    )
    results["hostsScanned"] = len(hosts)
    save(results)

    if not hosts:
        L("No hosts found.")
        results["status"] = "done"
        save(results)
        return

    # ── Step 2 — Model Discovery ─────────────────────────────
    L("\nStep 2 — Model Discovery")
    targets          = cfg.get("target_models", [])
    exclude_patterns = cfg.get("exclude_model_patterns", [])
    large_threshold  = float(cfg.get("large_model_threshold_b", 100))

    candidates, host_other_models = discover_candidates(
        hosts,
        targets,
        exclude_model_patterns=exclude_patterns,
        large_threshold_b=large_threshold,
        log_fn=L,
    )
    L(f"\n{len(candidates)} matching model(s) found.")
    if host_other_models:
        L(f"  {len(host_other_models)} host(s) have additional non-scanned models")

    if not candidates:
        results["status"] = "done"
        save(results)
        return

    # ── Step 3 — Benchmark ───────────────────────────────────
    bc  = cfg.get("benchmark", {})
    bcl = cfg.get("benchmark_large", {})

    if bc.get("enabled", True):
        L("\nStep 3 — Benchmark")

        normal_count = sum(1 for c in candidates if not c.is_large)
        large_count  = sum(1 for c in candidates if c.is_large)
        L(f"  {normal_count} normal, {large_count} large candidates")

        for c in candidates:
            benchmark_candidate(
                c,
                prompt=bc.get("prompt", "Hallo, wie geht es dir?"),
                max_ttft=float(bc.get("max_ttft_seconds", 15.0)),
                min_tps=float(bc.get("min_tokens_per_second", 3.0)),
                min_response_len=int(bc.get("min_response_length", 20)),
                runs=int(bc.get("runs", 2)),
                max_ttft_large=float(bcl.get("max_ttft_seconds", 120.0)),
                min_tps_large=float(bcl.get("min_tokens_per_second", 0.3)),
                runs_large=int(bcl.get("runs", 1)),
                tool_call_test=bool(bc.get("tool_call_test", True)),
                priority_models=cfg.get("priority_models", []),
                log_fn=L,
            )
    else:
        for c in candidates:
            c.benchmark_ok = True

    # ── Step 4 — LiteLLM Registration ────────────────────────
    L("\nStep 4 — LiteLLM Registration")
    lc  = cfg["litellm"]
    mgr = LiteLLMManager(
        lc["base_url"],
        lc["master_key"],
        cfg.get("litellm_model_tag", "ollama-scout"),
        log_fn=L,
    )

    passing = [c for c in candidates if c.benchmark_ok]
    L(f"{len(passing)}/{len(candidates)} passed.")

    for c in passing:
        mgr.add_model(c)

    # ── Save results ─────────────────────────────────────────
    results["status"] = "done"
    results["log"]    = log
    results["hostOtherModels"] = host_other_models
    results["candidates"] = [
        {
            "id":            i + 1,
            "ip":            c.host.ip,
            "port":          c.host.port,
            "country":       c.host.country,
            "org":           c.host.org,
            "model":         c.model_name,
            "matched":       c.matched_target,
            "isLarge":       c.is_large,
            "isHoneypot":    c.is_honeypot_flag,
            "honeypotReason": c.honeypot_reason,
            # Metadata
            "contextWindow": c.context_window,
            "quantization":  c.quantization,
            "quantScore":    c.quant_score,
            "hasVision":     c.has_vision,
            "hasTools":      c.has_tools or bool(c.tool_call_ok),
            "toolCallOk":    c.tool_call_ok,
            "parameterSize": c.parameter_size,
            "availability":  c.availability,
            # Benchmark
            "ttft":          round(c.ttft, 3) if c.ttft else None,
            "tps":           round(c.tokens_per_second, 1) if c.tokens_per_second else None,
            "response":      c.response_text,
            "status":        "honeypot" if c.is_honeypot_flag else ("added" if c.benchmark_ok else "failed"),
            "failReason":    c.benchmark_error if not c.benchmark_ok else None,
            "litellmName":   c.litellm_model_name if c.benchmark_ok else None,
        }
        for i, c in enumerate(candidates)
    ]
    save(results)
    L("\nScan complete.")


if __name__ == "__main__":
    main()
