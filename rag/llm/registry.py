"""Picks which LLM providers to use and falls back between them.

    from rag.llm import generate
    reply = generate(system_prompt, user_message)
    reply.text, reply.provider   # provider is "none" if nothing answered

If the first provider errors or is rate limited, the next one in
``config.LLM_PROVIDERS`` is tried. A rate-limited provider is skipped for
``LLM_RATE_LIMIT_COOLDOWN_SECONDS`` so a demo never stalls on it.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import config
from rag.llm.base import LLMProvider, ProviderError, RateLimitError
from rag.llm.http_providers import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)

log = logging.getLogger("rag.llm")

PROVIDER_CLASSES = {
    "openai": OpenAICompatibleProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
}

_cooldown_until: dict[str, float] = {}
_instances: dict[str, LLMProvider] = {}


@dataclass
class Reply:
    text: str
    provider: str                      # e.g. "groq:openai/gpt-oss-120b", or "none"
    errors: list[str] = field(default_factory=list)

    @property
    def generated(self) -> bool:
        return self.provider != "none"


def get_provider(name: str) -> LLMProvider:
    if name not in _instances:
        settings = config.PROVIDER_SETTINGS.get(name)
        if settings is None:
            raise ValueError(f"Unknown LLM provider {name!r}. Known: {sorted(config.PROVIDER_SETTINGS)}")
        _instances[name] = PROVIDER_CLASSES[settings["kind"]](name, settings)
    return _instances[name]


def wanted_names() -> list[str]:
    raw = (config.LLM_PROVIDERS or "auto").lower()
    if raw == "none":
        return []
    if raw == "auto":
        names = []
        for name in config.AUTO_PROVIDER_ORDER:
            needs_env = config.PROVIDER_SETTINGS[name].get("auto_requires_env")
            if needs_env and not os.environ.get(needs_env):
                continue
            names.append(name)
        return names
    return [n.strip() for n in raw.split(",") if n.strip()]


def active_providers() -> list[LLMProvider]:
    """Providers that are listed AND look usable right now, in order."""
    result = []
    for name in wanted_names():
        try:
            provider = get_provider(name)
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        if provider.is_configured():
            result.append(provider)
    return result


def describe_chain() -> str:
    chain = [p.describe() for p in active_providers()]
    return " -> ".join(chain) if chain else "none"


def generate(system: str, user: str, on_attempt=None, only: list[str] | None = None) -> Reply:
    """on_attempt(provider_name, previous_error_or_None) is called before
    each provider is tried (the chat page shows it as progress).
    ``only``: try just these providers (configured ones, in this order)."""
    errors: list[str] = []
    providers = active_providers() if only is None else [
        get_provider(n) for n in only if n in config.PROVIDER_SETTINGS and get_provider(n).is_configured()]
    for provider in providers:
        if _cooldown_until.get(provider.name, 0) > time.time():
            errors.append(f"{provider.name}: skipped (cooling down after a rate limit)")
            continue
        if on_attempt:
            on_attempt(provider.name, errors[-1] if errors else None)
        try:
            text = provider.generate(system, user)
            return Reply(text=text, provider=provider.describe(), errors=errors)
        except RateLimitError as exc:
            _cooldown_until[provider.name] = time.time() + config.LLM_RATE_LIMIT_COOLDOWN_SECONDS
            log.warning("%s -- trying the next provider", exc)
            errors.append(str(exc))
        except ProviderError as exc:
            log.warning("%s -- trying the next provider", exc)
            errors.append(str(exc))
        except Exception as exc:  # never let one provider crash the whole answer
            log.exception("Unexpected error from %s", provider.name)
            errors.append(f"{provider.name}: {exc}")
    return Reply(text="", provider="none", errors=errors)


def reset_state():
    """Forget cached providers and cooldowns (used by tests)."""
    _cooldown_until.clear()
    _instances.clear()
