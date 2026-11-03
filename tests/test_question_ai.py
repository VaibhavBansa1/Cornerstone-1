from rag import question_ai
from rag.llm import Reply


def test_reads_json_and_never_asks_a_local_model(monkeypatch):
    seen = {}

    def fake_generate(system, user, on_attempt=None, only=None):
        seen["only"] = only
        return Reply(text='thinking... {"topic": "timetable", "semester": 5, "english": "5th sem timetable"}',
                     provider="groq_small:x")

    monkeypatch.setattr(question_ai.llm, "generate", fake_generate)
    monkeypatch.setattr(question_ai.config, "QUESTION_AI_PROVIDERS", ["groq_small", "groq"])
    assert question_ai.read("paanchve sem ka routine")["semester"] == 5
    assert "ollama" not in seen["only"]


def test_no_provider_or_no_json_means_rules_only(monkeypatch):
    monkeypatch.setattr(question_ai.llm, "generate", lambda *a, **k: Reply(text="", provider="none"))
    assert question_ai.read("x") is None
    monkeypatch.setattr(question_ai.llm, "generate", lambda *a, **k: Reply(text="sorry", provider="groq:x"))
    assert question_ai.read("x") is None
    monkeypatch.setattr(question_ai.config, "QUESTION_AI_PROVIDERS", [])
    assert question_ai.read("x") is None
