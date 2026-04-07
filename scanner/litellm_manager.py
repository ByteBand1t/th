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
        """Standard LiteLLM params for Ollama models — compatible with Paperclip."""
        return {
            "model":                  litellm_model,
            "api_base":               api_base,
            "custom_llm_provider":    "ollama_chat",
            "timeout":                "900",
            "input_cost_per_token":   0,
            "output_cost_per_token":  0,
            "drop_params":            True,
            "extra_body": {
                "think":   False,
                "num_ctx": 32768,
            },
            "use_in_pass_through":              False,
            "use_litellm_proxy":                False,
            "merge_reasoning_content_in_choices": False,
            "parallel_tool_calls":              False,
        }

    def add_model(self, c: ModelCandidate) -> bool:
        existing = self._existing()

        # 1. Individual instance
        self._register(
            model_name=c.litellm_model_name,
            litellm_params=self._litellm_params(c.litellm_model_string, c.host.base_url),
            existing=existing,
            description=f"ollama-scout | {c.host.ip}:{c.host.port} | {c.host.country} | {c.host.org}",
            ttft=c.ttft,
            tps=c.tokens_per_second,
            is_pool=False,
        )

        # 2. Pool group
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
            "model_name":    model_name,
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
            label = "Pool" if is_pool else "Added"
            self.log(f"  ✓ {label}: {model_name}")
            return True
