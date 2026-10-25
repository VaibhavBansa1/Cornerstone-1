"""Show how each retrieval stage ranks chunks for a question.

    python scripts/debug_retrieval.py "Who is the HOD of IT?" [needle]

``needle`` (optional) is text you expect in the answer; the script prints
where chunks containing it rank in the keyword list, the vector list and
the final result, so you can see which stage loses them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ingestion.dates import label  # noqa: E402
from rag.retrieval import expand_query, load_retriever  # noqa: E402


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Who is the HOD of IT department?"
    needle = sys.argv[2].lower() if len(sys.argv) > 2 else None
    r = load_retriever()
    n = config.CANDIDATES_PER_RETRIEVER
    kw = r.bm25.search(expand_query(query), 300)
    vec = r._vector_search(query, 300)
    print("query tokens:", expand_query(query))

    def show(name, hits):
        print(f"\n--- {name} (top 8 of {len(hits)}) ---")
        for rank, (i, s) in enumerate(hits[:8]):
            c = r.chunks[i]
            print(f"{rank:>3} {s:7.3f} {label(c.get('date')):>9}  {c['text'][:90]!r}")
        if needle:
            ranks = [rank for rank, (i, _) in enumerate(hits) if needle in r.chunks[i]["text"].lower()]
            print(f"   ranks containing {needle!r}: {ranks[:10]} (candidate cut-off is {n})")

    show("keyword / BM25", kw)
    show("vector", vec)
    results = r.search(query, top_k=config.TOP_K)
    print("\n--- final ---")
    for c in results:
        print(f"rel={c['relevance']:.3f} +new={c['recency_bonus']:.3f} {label(c.get('date')):>9} {c['text'][:90]!r}")


if __name__ == "__main__":
    main()
