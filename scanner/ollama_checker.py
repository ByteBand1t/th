from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Callable, Optional
import httpx
from shodan_scanner import OllamaHost

TAGS_TIMEOUT = 6.0
BENCH_TIMEOUT = 90.0


@dataclass
class ModelCandidate:
    host: OllamaHost
    model_name: str
    matched_target: str
    ttft: Optional[float] = None
    tokens_per_second: Optional[float] = None
    response_text: Optional[str] = None
    benchmark_ok: Optional[bool] = None
    benchmark_error: Optional[str] = None

    @property
    def litellm_model_string(self): return f"ollama/{self.model_name}"

    @property
    def litellm_model_name(self):
        safe = self.model_name.replace(":", "-").replace("/", "-")
        return f"{safe}@{self.host.ip.replace('.', '-')}"


def _get_tags(host):
    try:
        r = httpx.get(f"{host.base_url}/api/tags", timeout=TAGS_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        pass
    return None


def discover_candidates(hosts, target_models, log_fn=print):
    candidates = []
    for host in hosts:
        tags = _get_tags(host)
        if tags is None:
            log_fn(f"  ✗ {host} — unreachable")
            continue
        matched = 0
        for m in tags:
            name = m.get("name", "")
            for t in target_models:
                if t.lower() in name.lower():
                    candidates.append(ModelCandidate(host=host, model_name=name, matched_target=t))
                    log_fn(f"  ✓ {name} on {host}")
                    matched += 1
                    break
        if not matched:
            avail = [m.get("name","") for m in tags]
            log_fn(f"  – {host} — no targets (has: {', '.join(avail[:3])})")
    return candidates


def _stream(base_url, model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": True, "options": {"num_predict": 80}}
    t0 = time.perf_counter()
    ttft, tokens, text = None, 0, ""
    with httpx.stream("POST", f"{base_url}/api/generate", json=payload, timeout=BENCH_TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line: continue
            try: chunk = json.loads(line)
            except: continue
            frag = chunk.get("response", "")
            if frag:
                if ttft is None: ttft = time.perf_counter() - t0
                tokens += 1
                text += frag
            if chunk.get("done"): break
    elapsed = time.perf_counter() - t0
    return (ttft or elapsed), (tokens / elapsed if elapsed else 0), text


def benchmark_candidate(c, prompt, max_ttft, min_tps, min_response_len, runs=2, log_fn=print):
    log_fn(f"  Benchmarking {c.model_name} @ {c.host.ip}")
    ttfts, tpss, last = [], [], ""
    for i in range(runs):
        try:
            ttft, tps, text = _stream(c.host.base_url, c.model_name, prompt)
            ttfts.append(ttft); tpss.append(tps); last = text
            log_fn(f"    Run {i+1}: TTFT={ttft:.2f}s  TPS={tps:.1f}  chars={len(text)}")
        except Exception as e:
            c.benchmark_ok = False; c.benchmark_error = str(e)
            log_fn(f"    Error: {e}")
            return c
    avg_ttft = sum(ttfts) / len(ttfts)
    avg_tps  = sum(tpss)  / len(tpss)
    c.ttft, c.tokens_per_second, c.response_text = avg_ttft, avg_tps, last
    reasons = []
    if avg_ttft > max_ttft:       reasons.append(f"TTFT {avg_ttft:.1f}s > {max_ttft}s")
    if avg_tps  < min_tps:        reasons.append(f"TPS {avg_tps:.1f} < {min_tps}")
    if len(last) < min_response_len: reasons.append(f"response too short ({len(last)} chars)")
    if reasons:
        c.benchmark_ok = False; c.benchmark_error = "; ".join(reasons)
        log_fn(f"    ✗ FAIL: {c.benchmark_error}")
    else:
        c.benchmark_ok = True
        log_fn(f"    ✓ PASS  TTFT={avg_ttft:.2f}s  TPS={avg_tps:.1f}")
    return c
