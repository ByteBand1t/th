from __future__ import annotations
from typing import Callable
import httpx
from ollama_checker import ModelCandidate

TIMEOUT = 10.0


class LiteLLMManager:
    def __init__(self, base_url, master_key, tag="ollama-scout", log_fn=print):
        self.base_url = base_url.rstrip("/")
        self.headers  = {"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"}
        self.tag, self.log = tag, log_fn

    def _existing(self):
        try:
            r = httpx.get(f"{self.base_url}/model/info", headers=self.headers, timeout=TIMEOUT)
            r.raise_for_status()
            return {m.get("model_name","") for m in r.json().get("data",[])}
        except Exception as e:
            self.log(f"  LiteLLM error: {e}")
            return set()

    def add_model(self, c: ModelCandidate) -> bool:
        name = c.litellm_model_name
        if name in self._existing():
            self.log(f"  ⏭  {name} already registered")
            return False
        payload = {
            "model_name": name,
            "litellm_params": {"model": c.litellm_model_string, "api_base": c.host.base_url},
            "model_info": {
                "description": f"ollama-scout | {c.host.ip}:{c.host.port} | {c.host.country}",
                "tags": [self.tag],
                "ttft_avg": round(c.ttft or 0, 3),
                "tps_avg":  round(c.tokens_per_second or 0, 1),
            },
        }
        try:
            r = httpx.post(f"{self.base_url}/model/new", headers=self.headers, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            self.log(f"  ✓ Added: {name}")
            return True
        except Exception as e:
            self.log(f"  ✗ Error: {e}")
            return False
