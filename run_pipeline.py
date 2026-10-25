"""Orchestrates the full flow: crawl -> ingest -> embed & index.

    python run_pipeline.py              # crawl the site again, then rebuild
    python run_pipeline.py --skip-crawl # rebuild from already-downloaded data
                                        # (use after changing chunking or the
                                        #  embedding model)
Then query with:
    python -m rag.pipeline "your question"
"""
import argparse
import logging

from ingestion.extract import build_chunks, write_chunks
from rag.embed_index import build_index
from rag.embeddings import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-crawl", action="store_true", help="reuse data/raw instead of crawling again")
    args = parser.parse_args()

    if args.skip_crawl:
        log.info("Step 1/3: skipping crawl, reusing existing data/raw and manifest.")
    else:
        from scraper.crawler import ScopedCrawler

        log.info("Step 1/3: crawling (scope-limited, single-visit)...")
        ScopedCrawler().crawl()

    log.info("Step 2/3: extracting, chunking and dating text...")
    chunks = build_chunks()
    write_chunks(chunks)

    log.info("Step 3/3: embedding on %s & building FAISS index...", resolve_device())
    build_index(chunks)

    log.info('Done. Query with: python -m rag.pipeline "your question"')


if __name__ == "__main__":
    main()
