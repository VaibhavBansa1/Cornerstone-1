"""Ask a list of test questions and save answers + sources to a file.

    python scripts/ask_batch.py                # built-in question set
    python scripts/ask_batch.py questions.txt  # one question per line

Output goes to data/eval_answers.md -- read it to judge answer quality
after a change (new model, new chunking, new data).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ingestion.dates import label  # noqa: E402
from rag.pipeline import answer  # noqa: E402

DEFAULT_QUESTIONS = [
    "Who is the HOD of IT department?",
    "IT ka HOD kaun hai abhi?",
    "What is the timetable for B.Tech IT 5th semester?",
    "Who is the time table coordinator?",
    "What subjects are in the 3rd semester syllabus?",
    "What is the vision of the IT department?",
    "Who was the HOD of IT in 2024?",
    "What is the hostel fee?",  # not in the data: should say it doesn't know
]


def main():
    if len(sys.argv) > 1:
        questions = [q.strip() for q in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if q.strip()]
    else:
        questions = DEFAULT_QUESTIONS
    out = [f"# Test answers ({time.strftime('%Y-%m-%d %H:%M')})\n"]
    for q in questions:
        started = time.perf_counter()
        r = answer(q)
        secs = time.perf_counter() - started
        out.append(f"## {q}\n\n*via {r['provider']}, {secs:.1f}s*\n\n{r['answer'].strip()}\n")
        out.append("Sources: " + "; ".join(
            f"[{i}] {label(s.get('date'))} {s.get('title') or s['source_url']}" for i, s in enumerate(r["sources"], 1)
        ) + "\n")
        print(f"done ({secs:.0f}s): {q}", flush=True)
    path = config.DATA_DIR / "eval_answers.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("Saved to", path)


if __name__ == "__main__":
    main()
