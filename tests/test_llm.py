"""Provider fallback tests -- no network: providers are faked."""
import pytest

import config
from rag.llm import registry
from rag.llm.base import LLMProvider, ProviderError, RateLimitError
from rag.llm.http_providers import clean_reply


class FakeProvider(LLMProvider):
    def __init__(self, name, behaviour):
        super().__init__(name, {"model": "fake"})
        self.behaviour = behaviour
        self.calls = 0

    def is_configured(self):
        return True

    def generate(self, system, user):
        self.calls += 1
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


@pytest.fixture(autouse=True)
def clean_registry():
    registry.reset_state()
    yield
    registry.reset_state()


def _install(monkeypatch, providers, order):
    monkeypatch.setattr(config, "LLM_PROVIDERS", order)
    for p in providers:
        registry._instances[p.name] = p
    monkeypatch.setattr(config, "PROVIDER_SETTINGS", {p.name: {"kind": "openai"} for p in providers})


def test_falls_back_to_next_provider_on_rate_limit(monkeypatch):
    first = FakeProvider("openrouter", RateLimitError("429"))
    second = FakeProvider("groq", "answer from groq")
    _install(monkeypatch, [first, second], "openrouter,groq")

    reply = registry.generate("sys", "q")
    assert reply.text == "answer from groq"
    assert reply.provider == "groq:fake"
    assert reply.generated

    # The rate-limited provider is skipped while it cools down.
    registry.generate("sys", "q")
    assert first.calls == 1
    assert second.calls == 2


def test_all_failing_returns_none(monkeypatch):
    _install(monkeypatch, [FakeProvider("a", ProviderError("down")), FakeProvider("b", ValueError("bug"))], "a,b")
    reply = registry.generate("sys", "q")
    assert reply.provider == "none"
    assert not reply.generated
    assert len(reply.errors) == 2


def test_none_disables_generation(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDERS", "none")
    assert registry.generate("s", "q").provider == "none"


def test_auto_skips_providers_without_keys(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDERS", "auto")
    for env in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "GITHUB_MODELS_TOKEN", "GEMINI_API_KEY",
                "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_COMPAT_BASE_URL"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(registry.OllamaProvider, "is_configured", lambda self: False)
    # the Groq key also enables groq_small (same key, smaller model, its own daily allowance)
    assert [p.name for p in registry.active_providers()] == ["groq", "groq_small"]


def test_reasoning_tags_are_removed():
    assert clean_reply("<think>hmm let me see</think>\nThe HOD is X.") == "The HOD is X."


def test_dotenv_file_sets_missing_variables_only(tmp_path, monkeypatch):
    import config

    f = tmp_path / ".env"
    f.write_text('# comment\nOPENROUTER_API_KEY="sk-or-test"\nGROQ_API_KEY=from-file\n', encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    config._load_dotenv(f)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test"
    assert os.environ["GROQ_API_KEY"] == "from-env"
    monkeypatch.delenv("OPENROUTER_API_KEY")
