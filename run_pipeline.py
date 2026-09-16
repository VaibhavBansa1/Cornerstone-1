"""Orchestrates the full flow: crawl -> ingest -> embed & index.

Run once to (re)build the corpus and index from scratch:
    python run_pipeline.py
Then query with:
    python -m rag.pipeline "your question"
"""
import logging

from ingestion.extract import build_chunks, write_chunks
from rag.embed_index import build_index
from scraper.crawler import ScopedCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")


def main():
    log.info("Step 1/3: crawling (scope-limited, single-visit)...")
    ScopedCrawler().crawl()

    log.info("Step 2/3: extracting & chunking text...")
    chunks = build_chunks()
    write_chunks(chunks)

    log.info("Step 3/3: embedding & building FAISS index...")
    build_index(chunks)

    log.info("Done. Query with: python -m rag.pipeline \"your question\"")


if __name__ == "__main__":
    main()
