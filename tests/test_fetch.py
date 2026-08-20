import pytest

from p3.config import Source
from p3.fetch import OfflineMiss, PoliteFetcher
from p3.raw_store import RawStore

SRC = Source(name="t", host="example.invalid", force_encoding="cp1251")


def test_cache_hit_does_not_touch_network(tmp_path):
    store = RawStore(tmp_path)
    url = "https://example.invalid/doc"
    store.put(url, "норма".encode("cp1251"), source="t", encoding="cp1251")

    fetcher = PoliteFetcher(store, offline=True)
    result = fetcher.get(url, SRC)

    assert result.from_cache is True
    assert result.text == "норма"


def test_offline_miss_is_explicit(tmp_path):
    fetcher = PoliteFetcher(RawStore(tmp_path), offline=True)
    with pytest.raises(OfflineMiss):
        fetcher.get("https://example.invalid/absent", SRC)


def test_relative_url_resolved_against_source(tmp_path):
    store = RawStore(tmp_path)
    store.put("https://example.invalid/a/b", b"x", source="t", encoding="cp1251")

    fetcher = PoliteFetcher(store, offline=True)
    assert fetcher.get("/a/b", SRC).from_cache is True
