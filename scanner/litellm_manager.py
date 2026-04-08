from __future__ import annotations
from typing import Callable
import httpx
from ollama_checker import ModelCandidate

TIMEOUT = 10.0


class LiteLLMManager:
    def __init__(self, base_url, master_key, tag="ollama-scout", log_fn=print):
        self.base_url = base_url.rstrip("/")
        self.headers  = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type":  "application/json",
        }
        self.tag, self.log = tag, log_fn

    def _existing(self) -> set[str]:
        try:
            r = httpx.get(
                f"{self.base_url}/model/info",
                headers=self.headers, timeout=TIMEOUT,
            )
            r.raise_for_status()
            return {m.get("model_name", "") for m in r.json().get("data", [])}
        except Exception as e:
            self.log(f"  LiteLLM error: {e}")
            return set()

    def _litellm_params(self, c: ModelCandidate) -> dict:
        """Build LiteLLM params using model metadata from /api/show."""
        # Use actual context window from model metadata, capped at 131072
        ctx = min(c.context_window, 131072)
        # If context window is the default 2048 fallback, use a safer 32768
        if ctx <= 2048:
            ctx = 32768

        return {
            "model":                              c.litellm_model_string,
            "api_base":                           c.host.base_url,
            "custom_llm_provider":                "ollama_chat",
            "timeout":                            "900",
            "input_cost_per_token":               0,
            "output_cost_per_token":              0,
            "drop_params":                        True,
            "extra_body": {
                "think":   False,
                "num_ctx": ctx,
            },
            "use_in_pass_through":                False,
            "use_litellm_proxy":                  False,
            "merge_reasoning_content_in_choices": False,
            "parallel_tool_calls":                False,
        }

    def add_model(self, c: ModelCandidate) -> bool:
        existing = self._existing()

        # 1. Individual instance
        self._register(
            model_name=c.litellm_model_name,
            params=self._litellm_params(c),
            existing=existing,
            description=(
                f"ollama-scout | {c.host.ip}:{c.host.port} | "
                f"{c.host.country} | {c.host.org}"
            ),
            c=c,
            is_pool=False,
        )

        # 2. Pool group
        self._register(
            model_name=c.pool_name,
            params=self._litellm_params(c),
            existing=existing,
            description=f"Pool: {c.pool_name} — auto-managed by ollama-scout",
            c=c,
            is_pool=True,
        )
        return True

    def _register(
        self,
        model_name: str,
        params: dict,
        existing: set[str],
        description: str,
        c: ModelCandidate,
        is_pool: bool = False,
    ) -> bool:
        if model_name in existing and not is_pool:
            self.log(f"  ⏭  {model_name} already registered")
            return False

        tags = [self.tag, "pool" if is_pool else "instance"]
        if c.has_vision: tags.append("vision")
        if c.has_tools:  tags.append("tools")
        if c.is_large:   tags.append("large")

        payload = {
            "model_name":     model_name,
            "litellm_params": params,
            "model_info": {
                "description":   description,
                "tags":          tags,
                "ttft_avg":      round(c.ttft or 0, 3),
                "tps_avg":       round(c.tokens_per_second or 0, 1),
                "context_window": c.context_window,
                "quantization":  c.quantization,
                "quant_score":   c.quant_score,
                "has_vision":    c.has_vision,
                "has_tools":     c.has_tools,
                "availability":  c.availability,
                "parameter_size": c.parameter_size,
            },
        }
        try:
            r = httpx.post(
                f"{self.base_url}/model/new",
                headers=self.headers, json=payload, timeout=TIMEOUT,
            )
            r.raise_for_status()
            label = "Pool" if is_pool else "Added"
            self.log(
                f"  ✓ {label}: {model_name} "
                f"| ctx={c.context_window} | {c.quantization} | "
                f"avail={c.availability:.0%}"
            )
            return True
        except Exception as e:
            self.log(f"  ✗ Error registering {model_name}: {e}")
            return False

    def _get_all(self) -> list[dict]:
        try:
            r = httpx.get(
                f"{self.base_url}/model/info",
                headers=self.headers, timeout=TIMEOUT,
            )
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
