from __future__ import annotations
import json, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import httpx
from shodan_scanner import OllamaHost

TAGS_TIMEOUT  = 6.0
BENCH_TIMEOUT = 120.0

# Persistent blocklist for hosts that returned 401
BLOCKLIST_FILE = Path("/data/blocked_hosts.json")


def _load_blocklist() -> set[str]:
    if BLOCKLIST_FILE.exists():
        try:
            return set(json.loads(BLOCKLIST_FILE.read_text()))
        except Exception:
            pass
    return set()


def _add_to_blocklist(ip: str, port: int, reason: str):
    blocked = _load_blocklist()
    key = f"{ip}:{port}"
    blocked.add(key)
    BLOCKLIST_FILE.write_text(json.dumps(sorted(blocked), indent=2))


def _extract_size_b(model_name: str) -> float:
    m = re.search(r":(\d+(?:\.\d+)?)b", model_name.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    if m:
        return float(m.group(1))
    return 0.0


@dataclass
class ModelCandidate:
    host: OllamaHost
    model_name: str
    matched_target: str
    min_size_b: float = 0.0
    ttft: Optional[float] = None
    tokens_per_second: Optional[float] = None
    response_text: Optional[str] = None
    benchmark_ok: Optional[bool] = None
    benchmark_error: Optional[str] = None

    @property
    def litellm_model_string(self):
        return f"ollama/{self.model_name}"

    @property
    def litellm_model_name(self):
        safe = self.model_name.replace(":", "-").replace("/", "-")
        return f"{safe}@{self.host.ip.replace('.', '-')}"

    @property
    def pool_name(self) -> str:
        base = self.matched_target.replace(":", "-")
        size = _extract_size_b(self.model_name)
        if size > 0:
            return f"{base}-{int(size)}b-pool"
        return f"{base}-pool"


def _get_tags(host: OllamaHost) -> list[dict] | None:
    """Fetch /api/tags. Returns None if unreachable, empty list if 401."""
    blocked = _load_blocklist()
    if f"{host.ip}:{host.port}" in blocked:
        return None
    try:
        r = httpx.get(f"{host.base_url}/api/tags", timeout=TAGS_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("models", [])
        if r.status_code == 401:
            _add_to_blocklist(host.ip, host.port, "401 on /api/tags")
            return None
    except Exception:
        pass
    return None


def _check_generate_auth(host: OllamaHost, model: str) -> bool:
    """
    Quick probe: send a minimal generate request and check for 401.
    Returns True if accessible, False if blocked.
    """
    try:
        payload = {"model": model, "prompt": "Hi", "stream": False,
                   "options": {"num_predict": 1}}
        r = httpx.post(f"{host.base_url}/api/generate", json=payload, timeout=8.0)
        if r.status_code == 401:
            _add_to_blocklist(host.ip, host.port, "401 on /api/generate")
            return False
        return True
    except Exception:
        # Timeout or connection error — not a 401, try benchmark anyway
        return True


def discover_candidates(hosts, target_models, log_fn=print):
    """target_models: list of dicts or strings."""
    targets = []
    for t in target_models:
        if isinstance(t, str):
            targets.append({"name": t, "min_size_b": 0.0})
        else:
            targets.append({"name": t["name"], "min_size_b": float(t.get("min_size_b", 0))})

    blocked = _load_blocklist()
    candidates = []

    for host in hosts:
        key = f"{host.ip}:{host.port}"
        if key in blocked:
            log_fn(f"  ⊘ {host} — in blocklist (previously 401)")
            continue

        tags = _get_tags(host)
        if tags is None:
            log_fn(f"  ✗ {host} — unreachable or blocked")
            continue

        matched = 0
        for m in tags:
            name = m.get("name", "")
            for t in targets:
                if t["name"].lower() in name.lower():
                    size = _extract_size_b(name)
                    if size < t["min_size_b"]:
                        log_fn(f"  ⊘ {name} @ {host.ip} — too small ({size}B < {t['min_size_b']}B)")
                        break
                    candidates.append(ModelCandidate(
                        host=host, model_name=name,
                        matched_target=t["name"],
                        min_size_b=t["min_size_b"],
                    ))
                    log_fn(f"  ✓ {name} on {host} ({size}B)")
                    matched += 1
                    break

        if not matched:
            avail = [m.get("name", "") for m in tags]
            log_fn(f"  – {host} — no targets (has: {', '.join(avail[:3])})")

    log_fn(f"\n  Blocklist has {len(blocked)} host(s) — skipped automatically")
    return candidates


def _stream(base_url, model, prompt):
    payload = {
        "model": model, "prompt": prompt,
        "stream": True, "options": {"num_predict": 80},
    }
    t0 = time.perf_counter()
    ttft, tokens, text = None, 0, ""

    with httpx.stream("POST", f"{base_url}/api/generate",
                      json=payload, timeout=BENCH_TIMEOUT) as r:
        # Check for 401 before reading body
        if r.status_code == 401:
            raise PermissionError("401 Unauthorized")
        r.raise_for_status()
        for line in r.iter_lines():
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
                text += frag
            if chunk.get("done"):
                break

    elapsed = time.perf_counter() - t0
    return (ttft or elapsed), (tokens / elapsed if elapsed else 0), text


def benchmark_candidate(c, prompt, max_ttft, min_tps, min_response_len,
                        runs=2, log_fn=print):
    log_fn(f"  Benchmarking {c.model_name} @ {c.host.ip}")

    # Auth pre-check
    if not _check_generate_auth(c.host, c.model_name):
        c.benchmark_ok = False
        c.benchmark_error = "401 Unauthorized — added to blocklist"
        log_fn(f"    ✗ {c.benchmark_error}")
        return c

    ttfts, tpss, last = [], [], ""

    for i in range(runs):
        try:
            ttft, tps, text = _stream(c.host.base_url, c.model_name, prompt)
            ttfts.append(ttft)
            tpss.append(tps)
            last = text
            log_fn(f"    Run {i+1}: TTFT={ttft:.2f}s  TPS={tps:.1f}  chars={len(text)}")
        except PermissionError as e:
            # 401 during streaming — blocklist and abort
            _add_to_blocklist(c.host.ip, c.host.port, str(e))
            c.benchmark_ok = False
            c.benchmark_error = str(e)
            log_fn(f"    ✗ {e} — added to blocklist")
            return c
        except Exception as e:
            c.benchmark_ok = False
            c.benchmark_error = str(e)
            log_fn(f"    Error: {e}")
            return c

    avg_ttft = sum(ttfts) / len(ttfts)
    avg_tps  = sum(tpss)  / len(tpss)
    c.ttft, c.tokens_per_second, c.response_text = avg_ttft, avg_tps, last

    reasons = []
    if avg_ttft > max_ttft:          reasons.append(f"TTFT {avg_ttft:.1f}s > {max_ttft}s")
    if avg_tps  < min_tps:           reasons.append(f"TPS {avg_tps:.1f} < {min_tps}")
    if len(last) < min_response_len: reasons.append(f"response too short ({len(last)} chars)")

    if reasons:
        c.benchmark_ok = False
        c.benchmark_error = "; ".join(reasons)
        log_fn(f"    ✗ FAIL: {c.benchmark_error}")
    else:
        c.benchmark_ok = True
        log_fn(f"    ✓ PASS  TTFT={avg_ttft:.2f}s  TPS={avg_tps:.1f}")

    return c
