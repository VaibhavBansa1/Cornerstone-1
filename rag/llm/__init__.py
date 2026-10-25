"""Swappable LLM providers. Use ``generate(system, user)`` and nothing else."""
from rag.llm.base import LLMProvider, ProviderError, RateLimitError
from rag.llm.registry import Reply, active_providers, describe_chain, generate, get_provider

__all__ = [
    "LLMProvider",
    "ProviderError",
    "RateLimitError",
    "Reply",
    "active_providers",
    "describe_chain",
    "generate",
    "get_provider",
]
