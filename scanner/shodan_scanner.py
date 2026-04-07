from __future__ import annotations
import shodan
from dataclasses import dataclass, field
from typing import Callable, List


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


def discover_hosts(api_key, queries, max_results=50, exclude_orgs=None, log_fn=print):
    api = shodan.Shodan(api_key)
    hosts, seen = [], set()
    exclude_orgs = [e.lower() for e in (exclude_orgs or [])]

    if isinstance(queries, str):
        queries = [queries]

    for query in queries:
        log_fn(f"  Query: {query}")
        try:
            results = api.search(query, limit=max_results)
            found = 0
            for m in results.get("matches", []):
                ip   = m.get("ip_str", "")
                port = m.get("port", 11434)
                org  = m.get("org", "unknown")
                key  = f"{ip}:{port}"

                if key in seen:
                    continue

                if any(ex in org.lower() for ex in exclude_orgs):
                    log_fn(f"    ⊘ Skip {ip} ({org}) — excluded")
                    continue

                seen.add(key)
                hosts.append(OllamaHost(
                    ip=ip, port=port,
                    country=m.get("location", {}).get("country_code", "??"),
                    org=org,
                    hostnames=m.get("hostnames", []),
                ))
                found += 1

            log_fn(f"    → {results.get('total', 0)} total, {found} new unique hosts")

        except shodan.APIError as e:
            log_fn(f"    Shodan error: {e}")

    log_fn(f"  {len(hosts)} unique hosts total")
    return hosts
