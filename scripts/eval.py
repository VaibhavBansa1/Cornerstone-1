"""Ask the test questions, grade the answers, write a report.

    python scripts/eval.py              # all questions in scripts/eval_questions.json
    python scripts/eval.py lab1 out3    # only these ids

Each question says what a good answer must contain (or that the bot should
say it doesn't know). Grading is simple pattern matching, so read the
answers in the report too -- it is a quick check, not a judge.

Every run is saved to data/eval/<time>.json; the report
(data/eval_report.md) compares the new run with the previous one.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.pipeline import answer  # noqa: E402

QUESTIONS = Path(__file__).with_name("eval_questions.json")
RUNS_DIR = config.DATA_DIR / "eval"
REPORT = config.DATA_DIR / "eval_report.md"

# How the bot says "I don't know" (English and Hinglish).
UNKNOWN = re.compile(
    r"not (explicitly |specifically )?(in|mentioned|available|found|covered|listed|given|specified)"
    r"|(do not|don't|does not|doesn't) (explicitly |specifically )?(have|contain|mention|include|cover|specify|list)"
    r"|(no|not any) (information|details|mention|data)"
    r"|could ?n[o']t find|cannot find|can't find|unable to find|not able to find"
    r"|i (don't|do not) know|not sure"
    r"|(jaankari|jankari|information) nahi|nahi (hai|mili|pata)|pata nahi",
    re.I,
)


def grade(item: dict, text: str) -> tuple[bool, str]:
    def found(p):
        return re.search(p, text, re.I)

    bad = [p for p in item.get("not", []) if found(p)]
    if bad:
        return False, f"says something it shouldn't: /{bad[0]}/"
    missing = [p for p in item.get("must", []) if not found(p)]
    hits = [p for p in item.get("any", []) if found(p)]
    need = item.get("min_any", 1) if item.get("any") else 0
    content_ok = not missing and len(hits) >= need
    says_unknown = bool(UNKNOWN.search(text))
    expect = item["expect"]
    if expect == "unknown":
        return (True, "says it doesn't know") if says_unknown else (False, "should have said it doesn't know")
    if expect == "either" and says_unknown and not item.get("must"):
        return True, "says it doesn't know (acceptable)"
    if content_ok:
        return True, "ok"
    if missing:
        return False, f"missing /{missing[0]}/"
    return False, f"only {len(hits)} of {need} expected items"


def main():
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    only = set(sys.argv[1:])
    if only:
        data = [q for q in data if q["id"] in only]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    previous = sorted(RUNS_DIR.glob("*.json"))
    before = {}
    if previous:
        before = {r["id"]: r for r in json.loads(previous[-1].read_text(encoding="utf-8"))["results"]}

    results = []
    for item in data:
        started = time.perf_counter()
        r = answer(item["q"])
        secs = time.perf_counter() - started
        ok, why = grade(item, r["answer"])
        results.append({
            "id": item["id"], "cat": item["cat"], "q": item["q"], "expect": item["expect"],
            "pass": ok, "why": why, "secs": round(secs, 1), "provider": r["provider"],
            "understood_as": r["understood_as"], "answer": r["answer"].strip(),
            "sources": [s.get("title") or s["source_url"] for s in r["sources"]],
        })
        print(f"{'PASS' if ok else 'FAIL'}  {secs:5.1f}s  {item['id']:6s} {item['q']}  ({why})", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RUNS_DIR / f"{stamp}.json").write_text(json.dumps({"time": stamp, "results": results}, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    write_report(results, before, stamp)
    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} passed. Report: {REPORT}")


def write_report(results, before, stamp):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["cat"]].append(r)
    total = sum(r["pass"] for r in results)
    out = [f"# Test questions: {total}/{len(results)} passed ({stamp})\n",
           "Automatic grading by keywords: read the answers below too.\n",
           "| Category | Passed | Before |", "|---|---|---|"]
    for cat, rs in by_cat.items():
        prev = [before[r["id"]]["pass"] for r in rs if r["id"] in before]
        prev_text = f"{sum(prev)}/{len(prev)}" if prev else "-"
        out.append(f"| {cat} | {sum(r['pass'] for r in rs)}/{len(rs)} | {prev_text} |")
    out.append(f"| **Total** | **{total}/{len(results)}** | |\n")
    for cat, rs in by_cat.items():
        out.append(f"## {cat}\n")
        for r in rs:
            mark = "PASS" if r["pass"] else "FAIL"
            out.append(f"### {mark} · {r['q']}\n")
            out.append(f"*expect: {r['expect']} · {r['why']} · {r['secs']}s via {r['provider']}"
                       + (f" · understood as: {r['understood_as']}" if r["understood_as"] else "") + "*\n")
            out.append(r["answer"] + "\n")
            out.append("Sources: " + "; ".join(f"[{i}] {s}" for i, s in enumerate(r["sources"], 1)) + "\n")
    REPORT.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
