"""LLM providers that talk plain HTTP -- no vendor SDKs to install.

* ``OpenAICompatibleProvider`` covers OpenRouter, Groq, GitHub Models,
  OpenAI and local servers like LM Studio, because they all speak the same
  "/chat/completions" format. Only the URL, key and model name differ.
* ``OllamaProvider`` uses Ollama's own local API.
* ``GeminiProvider`` and ``AnthropicProvider`` use their REST APIs.
"""
from __future__ import annotations

import logging
import re

import requests

import config
from rag.llm.base import LLMProvider, ProviderError, RateLimitError

log = logging.getLogger("rag.llm")

# Reasoning models (DeepSeek-R1, Qwen3, ...) wrap their thinking in these
# tags; students should only see the final answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def clean_reply(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _post(url: str, *, headers: dict, payload: dict, provider: str, timeout: int | None = None) -> dict:
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout or config.LLM_TIMEOUT_SECONDS)
    except requests.Timeout as exc:
        raise ProviderError(f"{provider}: no reply within {timeout or config.LLM_TIMEOUT_SECONDS}s") from exc
    except requests.RequestException as exc:
        raise ProviderError(f"{provider}: could not connect ({exc.__class__.__name__})") from exc
    if resp.status_code == 429:
        raise RateLimitError(f"{provider}: rate limited (HTTP 429)")
    if resp.status_code >= 400:
        raise ProviderError(f"{provider}: HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError(f"{provider}: response was not JSON") from exc


class OpenAICompatibleProvider(LLMProvider):
    def generate(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.name == "openrouter":  # optional; shows the app name on openrouter.ai
            headers["X-Title"] = "MITS IT Help Desk"
        data = _post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            payload={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": config.LLM_MAX_TOKENS,
                "temperature": config.LLM_TEMPERATURE,
            },
            provider=self.name,
        )
        # Some providers return 200 with an error body (e.g. upstream limit).
        if "error" in data and not data.get("choices"):
            message = str(data["error"])
            if "rate" in message.lower() or "429" in message:
                raise RateLimitError(f"{self.name}: {message[:200]}")
            raise ProviderError(f"{self.name}: {message[:300]}")
        try:
            text = clean_reply(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: unexpected response shape") from exc
        if not text:
            raise ProviderError(f"{self.name}: empty reply")
        return text


class OllamaProvider(LLMProvider):
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)
        self._resolved_model: str | None = None

    def is_configured(self) -> bool:
        try:
            requests.get(self.base_url, timeout=2)
            return True
        except requests.RequestException:
            return False

    def _pick_model(self) -> str:
        """Use the configured model, or the first one that is pulled."""
        if self._resolved_model:
            return self._resolved_model
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=5).json()
            installed = [m["name"] for m in tags.get("models", [])]
        except (requests.RequestException, ValueError, KeyError):
            installed = []
        wanted = self.model
        if installed and wanted not in installed and f"{wanted}:latest" not in installed:
            log.info("Ollama model %r not pulled; using %r instead.", wanted, installed[0])
            wanted = installed[0]
        self._resolved_model = wanted
        return wanted

    def describe(self) -> str:
        return f"{self.name}:{self._resolved_model or self.model}"

    def generate(self, system: str, user: str) -> str:
        data = _post(
            f"{self.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            payload={
                "model": self._pick_model(),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "keep_alive": self.settings.get("keep_alive", "30m"),
                "options": {"temperature": config.LLM_TEMPERATURE},
            },
            provider=self.name,
            timeout=self.settings.get("timeout"),
        )
        text = clean_reply((data.get("message") or {}).get("content", ""))
        if not text:
            raise ProviderError("ollama: empty reply")
        return text


class GeminiProvider(LLMProvider):
    def generate(self, system: str, user: str) -> str:
        data = _post(
            f"{self.base_url}/models/{self.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            payload={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "maxOutputTokens": config.LLM_MAX_TOKENS,
                    "temperature": config.LLM_TEMPERATURE,
                },
            },
            provider=self.name,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = clean_reply("".join(p.get("text", "") for p in parts))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("gemini: unexpected response shape") from exc
        if not text:
            raise ProviderError("gemini: empty reply")
        return text


class AnthropicProvider(LLMProvider):
    def generate(self, system: str, user: str) -> str:
        data = _post(
            f"{self.base_url}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            payload={
                "model": self.model,
                "max_tokens": config.LLM_MAX_TOKENS,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            provider=self.name,
        )
        text = clean_reply(
            "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        )
        if not text:
            raise ProviderError("anthropic: empty reply")
        return text
