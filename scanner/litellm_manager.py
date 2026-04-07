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
            return {m.get("model_name", "") for m in r.json().get("data", [])}
        except Exception as e:
            self.log(f"  LiteLLM error: {e}")
            return set()

    def _litellm_params(self, litellm_model: str, api_base: str) -> dict:
        return {
            "model":                              litellm_model,
            "api_base":                           api_base,
            "custom_llm_provider":                "ollama_chat",
            "timeout":                            "900",
            "input_cost_per_token":               0,
            "output_cost_per_token":              0,
            "drop_params":                        True,
            "extra_body":                         {"think": False, "num_ctx": 32768},
            "use_in_pass_through":                False,
            "use_litellm_proxy":                  False,
            "merge_reasoning_content_in_choices": False,
            "parallel_tool_calls":                False,
        }

    def add_model(self, c) -> bool:
        existing = self._existing()
        self._register(
            model_name=c.litellm_model_name,
            litellm_params=self._litellm_params(c.litellm_model_string, c.host.base_url),
            existing=existing,
            description=f"ollama-scout | {c.host.ip}:{c.host.port} | {c.host.country} | {c.host.org}",
            ttft=c.ttft,
            tps=c.tokens_per_second,
            is_pool=False,
        )
        self._register(
            model_name=c.pool_name,
            litellm_params=self._litellm_params(c.litellm_model_string, c.host.base_url),
            existing=existing,
            description=f"Pool: {c.pool_name} — auto-managed by ollama-scout",
            ttft=c.ttft,
            tps=c.tokens_per_second,
            is_pool=True,
        )
        return True

    def _register(self, model_name, litellm_params, existing,
                  description, ttft, tps, is_pool=False) -> bool:
        if model_name in existing and not is_pool:
            self.log(f"  ⏭  {model_name} already registered")
            return False
        payload = {
            "model_name":     model_name,
            "litellm_params": litellm_params,
            "model_info": {
                "description": description,
                "tags":        [self.tag, "pool" if is_pool else "instance"],
                "ttft_avg":    round(ttft or 0, 3),
                "tps_avg":     round(tps  or 0, 1),
            },
        }
        try:
            r = httpx.post(
                f"{self.base_url}/model/new",
                headers=self.headers, json=payload, timeout=TIMEOUT,
            )
            r.raise_for_status()
            self.log(f"  ✓ {'Pool' if is_pool else 'Added'}: {model_name}")
            return True
        except Exception as e:
            self.log(f"  ✗ Error registering {model_name}: {e}")
            return False

    def _get_all(self) -> list[dict]:
        try:
            r = httpx.get(f"{self.base_url}/model/info", headers=self.headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception:
            return []

    def remove_scout_models(self) -> int:
        removed = 0
        for model in self._get_all():
            if self.tag in model.get("model_info", {}).get("tags", []):
                model_id   = model.get("model_info", {}).get("id", "")
                model_name = model.get("model_name", "")
                try:
                    r = httpx.post(
                        f"{self.base_url}/model/delete",
                        headers=self.headers,
                        json={"id": model_id}, timeout=TIMEOUT,
                    )
                    r.raise_for_status()
                    self.log(f"  Removed: {model_name}")
                    removed += 1
                except Exception as e:
                    self.log(f"  Error removing {model_name}: {e}")
        return removed
