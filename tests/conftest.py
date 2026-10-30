
import pytest


@pytest.fixture(autouse=True)
def _no_college_pages(monkeypatch):
    """Crawler tests use a fake site; the real college page list would only
    add 404s. Tests that need it set config.COLLEGE_PAGES themselves."""
    import config

    monkeypatch.setattr(config, "COLLEGE_PAGES", [])
