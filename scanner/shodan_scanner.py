from __future__ import annotations
import json
import shodan
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

CACHE_FILE = Path("/data/known_hosts.json")
HOST_TTL_DAYS = 3       # Re-check known hosts after 3 days
MIN_CREDITS   = 2       # Stop scanning if fewer credits remain


@dataclass
class OllamaHost:
    ip: str
    port: int
    country: str = "??"
    org: str = "unknown"
    hostnames: list[str] = field(default_factory=list)
    from_cache: bool = False

    @property
    def base_url(self): return f"http://{self.ip}:{self.port}"
    def __str__(self): return f"{self.ip}:{self.port} [{self.country}/{self.org}]"


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


def discover_hosts(api_key, queries, max_results=50, exclude_orgs=None, log_fn=print) -> list[OllamaHost]:
    api = shodan.Shodan(api_key)
    exclude_orgs = [e.lower() for e in (exclude_orgs or [])]
    if isinstance(queries, str):
        queries = [queries]

    # Load cache
    cache = _load_cache()
    now   = datetime.now(timezone.utc)
    ttl   = timedelta(days=HOST_TTL_DAYS)

    # Separate fresh (skip Shodan) vs stale (re-query) cache entries
    fresh_hosts: list[OllamaHost] = []
    stale_keys:  set[str]         = set()

    for key, entry in cache.items():
        last_seen = datetime.fromisoformat(entry["last_seen"])
        host = OllamaHost(
            ip=entry["ip"], port=entry["port"],
            country=entry.get("country", "??"),
            org=entry.get("org", "unknown"),
            from_cache=True,
        )
        if now - last_seen < ttl:
            fresh_hosts.append(host)
        else:
            stale_keys.add(key)

    log_fn(f"  Cache: {len(fresh_hosts)} fresh hosts (skip), {len(stale_keys)} stale (re-check)")

    # Check remaining credits
    try:
        info    = api.info()
        credits = info.get("query_credits", 0)
        log_fn(f"  Shodan credits remaining: {credits}")
        if credits < MIN_CREDITS:
            log_fn(f"  ⚠ Less than {MIN_CREDITS} credits left — skipping Shodan queries, using cache only")
            return fresh_hosts
    except Exception as e:
        log_fn(f"  Could not check credits: {e}")

    # Shodan queries for new + stale hosts
    seen:  set[str]         = {_cache_key(h.ip, h.port) for h in fresh_hosts}
    new_hosts: list[OllamaHost] = []

    for query in queries:
        log_fn(f"  Query: {query}")
        try:
            results = api.search(query, limit=max_results)
            found   = 0
            for m in results.get("matches", []):
                ip   = m.get("ip_str", "")
                port = m.get("port", 11434)
                org  = m.get("org", "unknown")
                key  = _cache_key(ip, port)

                if key in seen:
                    # Update last_seen for stale entries
                    if key in stale_keys:
                        cache[key]["last_seen"] = now.isoformat()
                        stale_keys.discard(key)
                    continue

                if any(ex in org.lower() for ex in exclude_orgs):
                    log_fn(f"    ⊘ Skip {ip} ({org}) — excluded")
                    continue

                seen.add(key)
                host = OllamaHost(
                    ip=ip, port=port,
                    country=m.get("location", {}).get("country_code", "??"),
                    org=org,
                    hostnames=m.get("hostnames", []),
                    from_cache=False,
                )
                new_hosts.append(host)

                # Add to cache
                cache[key] = {
                    "ip": ip, "port": port,
                    "org": org,
                    "country": m.get("location", {}).get("country_code", "??"),
                    "last_seen": now.isoformat(),
                    "first_seen": now.isoformat(),
                }
                found += 1

            log_fn(f"    → {results.get('total', 0)} total, {found} new hosts")

        except shodan.APIError as e:
            log_fn(f"    Shodan error: {e}")

    _save_cache(cache)

    all_hosts = fresh_hosts + new_hosts
    log_fn(f"  Total: {len(all_hosts)} hosts ({len(fresh_hosts)} from cache, {len(new_hosts)} new from Shodan)")
    return all_hosts
