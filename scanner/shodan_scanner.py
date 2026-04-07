from __future__ import annotations
import shodan
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class OllamaHost:
    ip: str
    port: int
    country: str = "??"
    org: str = "unknown"
    hostnames: list[str] = field(default_factory=list)

    @property
    def base_url(self): return f"http://{self.ip}:{self.port}"
    def __str__(self): return f"{self.ip}:{self.port} [{self.country}/{self.org}]"


def discover_hosts(api_key, query, max_results=100, log_fn=print):
    api = shodan.Shodan(api_key)
    hosts, seen = [], set()
    log_fn(f"  Query: {query}")
    try:
        results = api.search(query, limit=max_results)
        log_fn(f"  {results.get('total',0)} total results, scanning up to {max_results}")
        for m in results.get("matches", []):
            ip, port = m.get("ip_str",""), m.get("port", 11434)
            key = f"{ip}:{port}"
            if key in seen: continue
            seen.add(key)
            hosts.append(OllamaHost(
                ip=ip, port=port,
                country=m.get("location",{}).get("country_code","??"),
                org=m.get("org","unknown"),
                hostnames=m.get("hostnames",[]),
            ))
    except shodan.APIError as e:
        log_fn(f"  Shodan error: {e}")
    log_fn(f"  {len(hosts)} unique hosts")
    return hosts
