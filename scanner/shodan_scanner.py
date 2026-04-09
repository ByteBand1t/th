"""shodan_scanner.py — discovers Ollama hosts via Shodan API"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import shodan

CACHE_FILE    = Path("/data/shodan_cache.json")
CACHE_TTL_SEC = 7 * 24 * 3600   # 7 days


@dataclass
class OllamaHost:
    ip: str
    port: int
    country: str = "??"
    org: str = "unknown"
    hostnames: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:{self.port}"

    def __str__(self) -> str:
        hn = self.hostnames[0] if self.hostnames else self.ip
        return f"{hn}:{self.port} [{self.country}/{self.org}]"


# ── Cache helpers ─────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _cache_key(ip: str, port: int) -> str:
    return f"{ip}:{port}"


# ── Main discovery ────────────────────────────────────────────

def discover_hosts(
    api_key: str,
    queries,
    max_results: int = 100,
    exclude_orgs: list[str] = None,
    exclude_ips: list[str] = None,
    log_fn: Callable = print,
) -> list[OllamaHost]:
    """
    Run multiple Shodan queries, deduplicate, cache results for 7 days.
    Reads port directly from Shodan result — works for any port.
    """
    api         = shodan.Shodan(api_key)
    cache       = _load_cache()
    now         = time.time()
    exclude_orgs = [o.lower() for o in (exclude_orgs or [])]
    exclude_ips  = set(exclude_ips or [])

    # Normalise queries to list
    if isinstance(queries, str):
        queries = [queries]

    seen:  set[str]         = set()
    hosts: list[OllamaHost] = []

    # ── Inject cached hosts first ────────────────────────────
    fresh, stale = 0, 0
    for key, entry in cache.items():
        age = now - entry.get("ts", 0)
        if age < CACHE_TTL_SEC:
            ip   = entry["ip"]
            port = entry["port"]
            if ip in exclude_ips:
                continue
            if any(ex in entry.get("org", "").lower() for ex in exclude_orgs):
                continue
            if key not in seen:
                seen.add(key)
                hosts.append(OllamaHost(
                    ip=ip, port=port,
                    country=entry.get("country", "??"),
                    org=entry.get("org", "unknown"),
                    hostnames=entry.get("hostnames", []),
                ))
                fresh += 1
        else:
            stale += 1

    log_fn(f"  Cache: {fresh} fresh (skip Shodan), {stale} stale (re-check)")

    # ── Shodan credits remaining ──────────────────────────────
    try:
        info = api.info()
        log_fn(f"  Shodan credits remaining: {info.get('unlocked_left', '?')}")
    except Exception:
        pass

    new_entries = {}

    for query in queries:
        log_fn(f"  Query: {query}")
        try:
            results = api.search(query, limit=max_results)
            total   = results.get("total", 0)
            added   = 0

            for match in results.get("matches", []):
                ip   = match.get("ip_str", "")
                port = match.get("port", 11434)
                org  = match.get("org", "unknown")
                key  = _cache_key(ip, port)

                # Skip excluded
                if ip in exclude_ips:
                    continue
                if any(ex in org.lower() for ex in exclude_orgs):
                    log_fn(f"  ⊘ Skip {ip} ({org}) — excluded")
                    continue
                if key in seen:
                    continue

                seen.add(key)
                added += 1
                host = OllamaHost(
                    ip=ip, port=port,
                    country=match.get("location", {}).get("country_code", "??"),
                    org=org,
                    hostnames=match.get("hostnames", []),
                )
                hosts.append(host)
                # Store in cache
                new_entries[key] = {
                    "ip":        ip,
                    "port":      port,
                    "country":   host.country,
                    "org":       org,
                    "hostnames": host.hostnames,
                    "ts":        now,
                }

            log_fn(f"    → {total} total, {added} new hosts")

        except shodan.APIError as e:
            log_fn(f"    ⊘ Shodan error for '{query}': {e}")
        except Exception as e:
            log_fn(f"    ⊘ Error for '{query}': {e}")

    # Update cache with new entries
    if new_entries:
        cache.update(new_entries)
        _save_cache(cache)

    log_fn(f"  Total: {len(hosts)} hosts ({fresh} cache, {len(hosts)-fresh} new)")
    return hosts
