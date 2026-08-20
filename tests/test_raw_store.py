import json

from p3.raw_store import RawStore, sha256_bytes


def test_put_get_roundtrip(tmp_path):
    store = RawStore(tmp_path)
    payload = "статья 88".encode("cp1251")
    entry = store.put("http://example/1", payload, source="t", encoding="cp1251")

    assert entry.sha256 == sha256_bytes(payload)
    assert entry.size == len(payload)
    assert store.get(entry.sha256) == payload
    assert store.get_text(entry) == "статья 88"


def test_manifest_survives_reload(tmp_path):
    RawStore(tmp_path).put("http://example/1", b"x", source="t")
    reloaded = RawStore(tmp_path)

    assert len(reloaded) == 1
    assert reloaded.lookup("http://example/1", "t") is not None
    assert reloaded.lookup("http://example/1", "other") is None


def test_identical_refetch_does_not_grow_manifest(tmp_path):
    store = RawStore(tmp_path)
    store.put("http://example/1", b"same", source="t")
    store.put("http://example/1", b"same", source="t")

    lines = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_changed_content_appends_new_entry(tmp_path):
    store = RawStore(tmp_path)
    first = store.put("http://example/1", b"old", source="t")
    second = store.put("http://example/1", b"new", source="t")

    assert first.sha256 != second.sha256
    # Обе версии остаются на диске: сайт изменился, старое доказательство не теряем.
    assert store.get(first.sha256) == b"old"
    assert store.lookup("http://example/1", "t").sha256 == second.sha256

    lines = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[-1])["sha256"] == second.sha256


def test_lookup_ignores_entry_whose_bytes_are_gone(tmp_path):
    store = RawStore(tmp_path)
    entry = store.put("http://example/1", b"x", source="t")
    store.path_for(entry.sha256).unlink()

    assert store.lookup("http://example/1", "t") is None


def test_exit_country_is_recorded_as_provenance(tmp_path):
    """Источник может отдавать разное разным адресам — происхождение должно быть видно."""
    store = RawStore(tmp_path)
    entry = store.put("http://example/1", b"x", source="t", exit_country="FI")

    assert entry.exit_country == "FI"
    assert RawStore(tmp_path).lookup("http://example/1", "t").exit_country == "FI"


def test_manifest_written_before_provenance_still_loads(tmp_path):
    """Старые строки манифеста не должны ломать загрузку после расширения схемы."""
    (tmp_path / "manifest.jsonl").write_text(
        '{"sha256":"a","url":"u","source":"t","fetched_at":"2026-01-01T00:00:00+00:00"}\n',
        encoding="utf-8",
    )
    store = RawStore(tmp_path)
    assert len(store) == 1
