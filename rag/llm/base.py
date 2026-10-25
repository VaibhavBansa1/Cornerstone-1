"""The one interface every LLM provider follows.

The rest of the app only ever calls ``provider.generate(system, user)``. To
add a new provider, subclass ``LLMProvider`` in its own file and register it
in ``registry.py`` -- nothing else in the project needs to change.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class ProviderError(Exception):
    """The provider failed (bad key, model missing, server down, ...)."""


class RateLimitError(ProviderError):
    """The provider said 'too many requests' -- try another one for a while."""


class LLMProvider(ABC):
    def __init__(self, name: str, settings: dict):
        self.name = name
        self.settings = settings
        self.model = settings.get("model", "")
        self.base_url = settings.get("base_url", "").rstrip("/")

    @property
    def api_key(self) -> str:
        for key in ("api_key_env", "alt_api_key_env"):
            env_name = self.settings.get(key)
            if env_name and os.environ.get(env_name, "").strip():
                return os.environ[env_name].strip()
        return ""

    def is_configured(self) -> bool:
        """Cheap check: does it look usable (key present, server up)?"""
        return bool(self.api_key) or self.settings.get("key_optional", False)

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the model's reply text, or raise ProviderError."""

    def describe(self) -> str:
        return f"{self.name}:{self.model}"
