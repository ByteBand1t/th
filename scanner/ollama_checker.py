from __future__ import annotations
import json, re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import httpx
from shodan_scanner import OllamaHost

TAGS_TIMEOUT        = 4.0
SHOW_TIMEOUT        = 6.0
BENCH_TIMEOUT       = 45.0
BENCH_TIMEOUT_LARGE = 300.0
AUTH_TIMEOUT        = 8.0
TOOL_TIMEOUT        = 20.0

BLOCKLIST_FILE   = Path("/data/blocked_hosts.json")
AVAIL_FILE       = Path("/data/availability.json")

# ── Quantization quality score (higher = better) ─────────────────
QUANT_SCORE: dict[str, int] = {
    "F32": 11, "F16": 10, "BF16": 10,
    "Q8_0": 9,
    "Q6_K": 8,
    "Q5_K_M": 7, "Q5_K_S": 7, "Q5_0": 6,
    "Q4_K_M": 6, "Q4_K_S": 5, "Q4_0": 5,
    "Q3_K_M": 4, "Q3_K_S": 4, "Q3_K_L": 4,
    "Q2_K": 2,
    "IQ4_XS": 5, "IQ4_NL": 5,
    "IQ3_M": 3, "IQ3_S": 3, "IQ3_XS": 3,
    "IQ2_M": 2, "IQ2_S": 2, "IQ2_XS": 2,
    "IQ1_S": 1, "IQ1_M": 1,
}

# Simple tool definition for tool-call test
TOOL_CALL_TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]
TOOL_CALL_TEST_PROMPT = "What's the weather in Berlin? Use the get_weather tool."


# ── Blocklist helpers ─────────────────────────────────────────────

def _load_blocklist() -> set[str]:
    if BLOCKLIST_FILE.exists():
        try:
            return set(json.loads(BLOCKLIST_FILE.read_text()))
        except Exception:
            pass
    return set()


def _add_to_blocklist(ip: str, port: int, reason: str):
    blocked = _load_blocklist()
    blocked.add(f"{ip}:{port}")
    BLOCKLIST_FILE.write_text(json.dumps(sorted(blocked), indent=2))


# ── Availability score helpers ────────────────────────────────────

def _load_avail() -> dict:
    if AVAIL_FILE.exists():
        try:
            return json.loads(AVAIL_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_avail(data: dict):
    AVAIL_FILE.write_text(json.dumps(data, indent=2))


def record_availability(model_key: str, alive: bool):
    """Record a health check result for availability scoring."""
    data = _load_avail()
    if model_key not in data:
        data[model_key] = {"checks": 0, "alive": 0}
    data[model_key]["checks"] += 1
    if alive:
        data[model_key]["alive"] += 1
    _save_avail(data)


def get_availability_score(model_key: str) -> float:
    """Returns availability as 0.0–1.0. Returns 1.0 if no history yet."""
    data = _load_avail()
    if model_key not in data or data[model_key]["checks"] == 0:
        return 1.0
    e = data[model_key]
    return round(e["alive"] / e["checks"], 3)


# ── Size extraction ───────────────────────────────────────────────

def _extract_size_b(model_name: str) -> float:
    name = model_name.lower()
    m = re.search(r"\d+x(\d+(?:\.\d+)?)b", name)
    if m:
        return float(m.group(1))
    m = re.search(r"[:\-](\d+(?:\.\d+)?)b", name)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)b", name)
    if m:
        return float(m.group(1))
    return 0.0


def _is_cloud_model(model_name: str, exclude_patterns: list[str]) -> bool:
    name_lower = model_name.lower()
    for p in exclude_patterns:
        if p.lower() in name_lower:
            return True
    return False


# ── /api/show metadata ────────────────────────────────────────────

def _get_model_info(host: OllamaHost, model_name: str) -> dict:
    """
    Call /api/show and return a dict with:
      context_window  int
      quantization    str
      quant_score     int
      capabilities    list[str]
      parameter_size  str
      has_vision      bool
      has_tools       bool
    """
    defaults = {
        "context_window": 2048,
        "quantization":   "unknown",
        "quant_score":    5,
        "capabilities":   [],
        "parameter_size": "?",
        "has_vision":     False,
        "has_tools":      False,
    }
    try:
        r = httpx.post(
            f"{host.base_url}/api/show",
            json={"model": model_name},
            timeout=SHOW_TIMEOUT,
        )
        if r.status_code != 200:
            return defaults
        data = r.json()

        # Context window — look in model_info first, then details
        mi   = data.get("model_info", {})
        ctx  = 2048
        for k, v in mi.items():
            if "context_length" in k:
                try:
                    ctx = int(v)
                    break
                except Exception:
                    pass

        # Quantization
        details = data.get("details", {})
        quant   = details.get("quantization_level", "unknown")
        qscore  = QUANT_SCORE.get(quant, 5)

        # Capabilities
        caps      = data.get("capabilities", [])
        has_vision = "vision" in caps
        has_tools  = "tools" in caps

        # Parameter size string
        param_size = details.get("parameter_size", "?")

        return {
            "context_window": ctx,
            "quantization":   quant,
            "quant_score":    qscore,
            "capabilities":   caps,
            "parameter_size": param_size,
            "has_vision":     has_vision,
            "has_tools":      has_tools,
        }
    except Exception:
        return defaults


# ── Model candidate dataclass ─────────────────────────────────────

@dataclass
class ModelCandidate:
    host: OllamaHost
    model_name: str
    matched_target: str
    min_size_b: float = 0.0
    is_large: bool = False
    # Metadata from /api/show
    context_window: int = 2048
    quantization: str = "unknown"
    quant_score: int = 5
    has_vision: bool = False
    has_tools: bool = False
    parameter_size: str = "?"
    # Benchmark results
    ttft: Optional[float] = None
    tokens_per_second: Optional[float] = None
    response_text: Optional[str] = None
    benchmark_ok: Optional[bool] = None
    benchmark_error: Optional[str] = None
    tool_call_ok: Optional[bool] = None
    # Availability
    availability: float = 1.0

    @property
    def model_key(self) -> str:
        return f"{self.model_name}@{self.host.ip}:{self.host.port}"

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


# ── Discovery ─────────────────────────────────────────────────────

def _get_tags(host: OllamaHost) -> list[dict] | None:
    blocked = _load_blocklist()
    if f"{host.ip}:{host.port}" in blocked:
        return None
    try:
        r = httpx.get(
            f"{host.base_url}/api/tags",
            timeout=httpx.Timeout(4.0, connect=2.0),
        )
        if r.status_code == 200:
            return r.json().get("models", [])
        if r.status_code == 401:
            _add_to_blocklist(host.ip, host.port, "401 on /api/tags")
            return None
    except Exception:
        pass
    return None


def _check_generate_auth(host: OllamaHost, model: str) -> bool:
    try:
        r = httpx.post(
            f"{host.base_url}/api/generate",
            json={"model": model, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}},
            timeout=AUTH_TIMEOUT,
        )
        if r.status_code == 401:
            _add_to_blocklist(host.ip, host.port, "401 on /api/generate")
            return False
        return True
    except Exception:
        return True


def discover_candidates(
    hosts,
    target_models,
    exclude_model_patterns: list[str] = None,
    large_threshold_b: float = 100.0,
    log_fn: Callable = print,
) -> list[ModelCandidate]:
    targets = []
    for t in target_models:
        if isinstance(t, str):
            targets.append({"name": t, "min_size_b": 0.0})
        else:
            targets.append({"name": t["name"], "min_size_b": float(t.get("min_size_b", 0))})

    exclude_patterns = exclude_model_patterns or []
    blocked          = _load_blocklist()
    candidates       = []

    for host in hosts:
        key = f"{host.ip}:{host.port}"
        if key in blocked:
            log_fn(f"  ⊘ {host} — in blocklist")
            continue

        tags = _get_tags(host)
        if tags is None:
            log_fn(f"  ✗ {host} — unreachable or blocked")
            continue

        matched = 0
        for m in tags:
            name = m.get("name", "")

            if _is_cloud_model(name, exclude_patterns):
                log_fn(f"  ⊘ {name} @ {host.ip} — cloud proxy, skipping")
                continue

            for t in targets:
                if t["name"].lower() in name.lower():
                    size = _extract_size_b(name)
                    if size < t["min_size_b"]:
                        log_fn(f"  ⊘ {name} @ {host.ip} — too small ({size}B < {t['min_size_b']}B)")
                        break

                    is_large = size >= large_threshold_b

                    # Fetch metadata from /api/show
                    info = _get_model_info(host, name)

                    # Load availability history
                    avail = get_availability_score(f"{name}@{host.ip}:{host.port}")

                    c = ModelCandidate(
                        host=host,
                        model_name=name,
                        matched_target=t["name"],
                        min_size_b=t["min_size_b"],
                        is_large=is_large,
                        context_window=info["context_window"],
                        quantization=info["quantization"],
                        quant_score=info["quant_score"],
                        has_vision=info["has_vision"],
                        has_tools=info["has_tools"],
                        parameter_size=info["parameter_size"],
                        availability=avail,
                    )
                    candidates.append(c)

                    badges = []
                    if is_large:      badges.append("LARGE")
                    if info["has_vision"]: badges.append("👁 vision")
                    if info["has_tools"]:  badges.append("🔧 tools")
                    badge_str = f" [{', '.join(badges)}]" if badges else ""

                    log_fn(
                        f"  ✓ {name} on {host} "
                        f"| ctx={info['context_window']} | {info['quantization']} (Q{info['quant_score']})"
                        f" | avail={avail:.0%}{badge_str}"
                    )
                    matched += 1
                    break

        if not matched:
            avail = [m.get("name", "") for m in tags]
            log_fn(f"  – {host} — no targets (has: {', '.join(avail[:3])})")

    normal = sum(1 for c in candidates if not c.is_large)
    large  = sum(1 for c in candidates if c.is_large)
    log_fn(f"\n  {len(candidates)} candidates: {normal} normal, {large} large (≥{large_threshold_b}B)")
    log_fn(f"  Blocklist: {len(blocked)} hosts skipped")
    return candidates


# ── Benchmark ─────────────────────────────────────────────────────

def _stream(base_url: str, model: str, prompt: str, timeout: float):
    payload = {
        "model": model, "prompt": prompt,
        "stream": True, "options": {"num_predict": 80},
    }
    t0 = time.perf_counter()
    ttft, tokens, text = None, 0, ""
    with httpx.stream("POST", f"{base_url}/api/generate",
                      json=payload, timeout=timeout) as r:
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


def _test_tool_call(base_url: str, model: str) -> bool:
    """Send a tool-call request and check if the model uses the tool correctly."""
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": TOOL_CALL_TEST_PROMPT}],
        "tools":    TOOL_CALL_TEST_TOOLS,
        "stream":   False,
    }
    try:
        r = httpx.post(f"{base_url}/api/chat", json=payload, timeout=TOOL_TIMEOUT)
        if r.status_code != 200:
            return False
        data   = r.json()
        msg    = data.get("message", {})
        calls  = msg.get("tool_calls", [])
        # Check that at least one tool call references our function
        for call in calls:
            fn = call.get("function", {})
            if fn.get("name") == "get_weather":
                return True
        return False
    except Exception:
        return False


def benchmark_candidate(
    c: ModelCandidate,
    prompt: str,
    max_ttft: float,
    min_tps: float,
    min_response_len: int,
    runs: int = 2,
    max_ttft_large: float = 120.0,
    min_tps_large: float = 0.3,
    runs_large: int = 1,
    tool_call_test: bool = True,
    log_fn: Callable = print,
) -> ModelCandidate:

    if c.is_large:
        eff_max_ttft = max_ttft_large
        eff_min_tps  = min_tps_large
        eff_runs     = runs_large
        eff_timeout  = BENCH_TIMEOUT_LARGE
        log_fn(f"  [LARGE] Benchmarking {c.model_name} @ {c.host.ip} "
               f"(TTFT≤{eff_max_ttft}s, TPS≥{eff_min_tps}, {eff_runs} run)")
    else:
        eff_max_ttft = max_ttft
        eff_min_tps  = min_tps
        eff_runs     = runs
        eff_timeout  = BENCH_TIMEOUT
        log_fn(f"  Benchmarking {c.model_name} @ {c.host.ip}")

    if not _check_generate_auth(c.host, c.model_name):
        c.benchmark_ok    = False
        c.benchmark_error = "401 Unauthorized — added to blocklist"
        log_fn(f"    ✗ {c.benchmark_error}")
        record_availability(c.model_key, False)
        return c

    ttfts, tpss, last = [], [], ""

    for i in range(eff_runs):
        try:
            ttft, tps, text = _stream(
                c.host.base_url, c.model_name, prompt, eff_timeout
            )
            ttfts.append(ttft)
            tpss.append(tps)
            last = text
            log_fn(f"    Run {i+1}: TTFT={ttft:.2f}s  TPS={tps:.1f}  chars={len(text)}")
        except PermissionError as e:
            _add_to_blocklist(c.host.ip, c.host.port, str(e))
            c.benchmark_ok    = False
            c.benchmark_error = str(e)
            log_fn(f"    ✗ {e}")
            record_availability(c.model_key, False)
            return c
        except Exception as e:
            c.benchmark_ok    = False
            c.benchmark_error = str(e)
            log_fn(f"    Error: {e}")
            record_availability(c.model_key, False)
            return c

    avg_ttft = sum(ttfts) / len(ttfts)
    avg_tps  = sum(tpss)  / len(tpss)
    c.ttft, c.tokens_per_second, c.response_text = avg_ttft, avg_tps, last

    reasons = []
    if avg_ttft > eff_max_ttft:          reasons.append(f"TTFT {avg_ttft:.1f}s > {eff_max_ttft}s")
    if avg_tps  < eff_min_tps:           reasons.append(f"TPS {avg_tps:.2f} < {eff_min_tps}")
    if len(last) < min_response_len:     reasons.append(f"response too short ({len(last)} chars)")

    if reasons:
        c.benchmark_ok    = False
        c.benchmark_error = "; ".join(reasons)
        log_fn(f"    ✗ FAIL: {c.benchmark_error}")
        record_availability(c.model_key, False)
    else:
        c.benchmark_ok = True
        log_fn(f"    ✓ PASS  TTFT={avg_ttft:.2f}s  TPS={avg_tps:.2f} "
               f"| Q{c.quant_score} ({c.quantization}) | ctx={c.context_window}")
        record_availability(c.model_key, True)

        # Tool-call test (only for passing models that declare tools capability
        # OR if tool_call_test is forced)
        if tool_call_test:
            log_fn(f"    🔧 Testing tool-call capability…")
            c.tool_call_ok = _test_tool_call(c.host.base_url, c.model_name)
            if c.tool_call_ok:
                log_fn(f"    ✓ Tool-call: YES")
                c.has_tools = True
            else:
                log_fn(f"    – Tool-call: NO (or not supported)")

    # Update availability score
    c.availability = get_availability_score(c.model_key)

    return c
