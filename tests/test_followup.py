"""Follow-up questions: rewritten into complete questions, nothing stored."""
from rag import followup, llm, pipeline


def test_history_is_cleaned_and_capped():
    raw = [{"q": "q1", "a": "a1"}, "junk", {"q": "", "a": "x"}, {"q": "q2", "a": "a" * 2000},
           {"q": "q3"}, {"q": "q4", "a": "a4"}]
    turns = followup.clean_history(raw)
    assert [t["q"] for t in turns] == ["q2", "q3", "q4"]
    assert len(turns[0]["a"]) == followup.MAX_ANSWER_CHARS


def test_no_history_means_no_ai_call(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert followup.standalone("Who is the HOD?", []) == "Who is the HOD?"


def test_follow_up_is_rewritten_with_the_conversation(monkeypatch):
    seen = {}

    def fake_generate(system, user, **_):
        seen["user"] = user
        return llm.Reply(text="Who was the HOD of IT before Dr. Punit Kumar Johari?\n", provider="fake")

    monkeypatch.setattr(llm, "generate", fake_generate)
    history = [{"q": "Who is the HOD of IT?", "a": "Dr. Punit Kumar Johari, Dean, School of IT."}]
    assert followup.standalone("and before him?", history) == "Who was the HOD of IT before Dr. Punit Kumar Johari?"
    assert "Dr. Punit Kumar Johari" in seen["user"] and "and before him?" in seen["user"]


def test_without_an_ai_the_previous_question_helps_the_search(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **k: llm.Reply(text="", provider="none"))
    assert followup.standalone("and 3rd sem?", [{"q": "5th sem syllabus", "a": "..."}]) == "5th sem syllabus and 3rd sem?"


def test_pipeline_searches_with_the_rewritten_question(monkeypatch):
    searched, steps = {}, []
    monkeypatch.setattr(followup, "standalone", lambda q, h: "3rd sem syllabus 2026")
    monkeypatch.setattr(pipeline, "retrieve", lambda q, **k: searched.setdefault("q", q) and [])

    class FakeRetriever:
        chunks = []

    monkeypatch.setattr(pipeline, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(pipeline, "generate_answer",
                        lambda q, r, **k: llm.Reply(text=f"answer for {q} (asked as {k['asked_as']})", provider="fake"))
    out = pipeline.answer("and 3rd sem?", history=[{"q": "5th sem syllabus 2026", "a": "..."}],
                          on_step=lambda kind, data: steps.append(kind))
    assert searched["q"] == "3rd sem syllabus 2026"
    assert out["rewritten"] == "3rd sem syllabus 2026" and out["query"] == "and 3rd sem?"
    assert "asked as and 3rd sem?" in out["answer"]
    assert steps[0] == "rewritten" and out["understood_as"].startswith("3rd sem")


def test_follow_up_prompt_keeps_the_students_own_words():
    _, user = pipeline.build_prompt("Who was the HOD before Dr. X?", [], asked_as="aur unse pehle?")
    assert "The student wrote: aur unse pehle?" in user and "Question (complete): Who was the HOD before Dr. X?" in user


def test_answers_to_english_letter_questions_are_asked_for_in_english_letters():
    from rag.pipeline import build_prompt, script_reminder

    assert "English letters" in script_reminder("canteen me kya kya milta hai?")
    assert script_reminder("कैंटीन में क्या मिलता है?") == ""
    system, user = build_prompt("canteen me kya milta hai?", [])
    assert user.rstrip().endswith("English or Hinglish.)")
    assert "classroom of that class" in system and "another department" in system


def test_rewrite_prompt_asks_for_a_question_and_merges_corrections():
    from rag.followup import REWRITE_PROMPT

    assert "QUESTION" in REWRITE_PROMPT and "time table" in REWRITE_PROMPT
