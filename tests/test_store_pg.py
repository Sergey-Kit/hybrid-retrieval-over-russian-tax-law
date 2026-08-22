"""Хранилище в PostgreSQL. Тесты на живом сервере, если он доступен.

`pgserver` поднимает PostgreSQL в пользовательском окружении, без root. Если
пакета нет — тесты пропускаются, а не падают: наличие СУБД не должно быть
условием прогона всего набора.
"""
import pytest

from p3.store.pg import PgChunkStore, corpus_build_id, schema_sql

pytest.importorskip("pgserver")
pytest.importorskip("psycopg")


def chunk(chunk_id, norm_unit_id, *, valid_from=None, valid_to=99991231,
          status="active", refs=(), part=1, article="88", clause="3"):
    return {
        "chunk_id": chunk_id, "norm_unit_id": norm_unit_id,
        "version_id": norm_unit_id + "@v1", "code": "НК РФ", "part": part,
        "section": "V", "chapter": "14", "article": article,
        "article_title": "Камеральная налоговая проверка", "clause": clause,
        "subclause": None, "valid_from": valid_from, "valid_to": valid_to,
        "applies_from": None, "published_at": None, "status": status,
        "version_no": 1, "amended_by": None, "source_type": "norm", "authority": 1.0,
        "refs_out": [{"target": t, "type": ty, "raw": ""} for t, ty in refs],
        "para_offsets": [[0, 10]], "text_dense": "префикс\n———\nтекст",
        "text_bm25": "текст нормы", "title_bm25": "заголовок",
    }


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    import pgserver
    import psycopg

    # XDG_RUNTIME_DIR подменяется в conftest — до импорта pgserver, который
    # вычисляет путь к файлу блокировки на импорте модуля.
    try:
        server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop")
    except Exception as exc:  # noqa: BLE001 — среда без работающего сервера
        pytest.skip(f"PostgreSQL не поднялся: {exc}")
    with psycopg.connect(server.get_uri()) as connection:
        item = PgChunkStore(connection)
        item.create_schema()
        connection.commit()
        yield item


def test_schema_avoids_pgvector():
    """Расширение не используется: на 8198 чанках ANN не нужен, а его бинарник
    падает с illegal instruction на процессоре без нужного набора инструкций."""
    sql = schema_sql()
    assert "CREATE EXTENSION" not in sql
    assert "REAL[]" in sql


def test_load_and_integrity(store):
    chunks = [chunk("a#0", "nk1/st88/p3"), chunk("b#0", "nk1/st88/p7", clause="7")]
    build = corpus_build_id(chunks)
    store.load_chunks(chunks, build)
    store.load_vectors(["a#0", "b#0"], [[0.1, 0.2], [0.3, 0.4]], "enc", build)
    store.connection.commit()

    report = store.integrity(build)
    assert report == {"chunks": 2, "vectors": 2, "chunks_without_vector": 0, "consistent": True}


def test_integrity_notices_a_missing_vector(store):
    chunks = [chunk("c#0", "nk1/st93/p1", article="93", clause="1")]
    build = corpus_build_id(chunks)
    store.load_chunks(chunks, build)
    store.connection.commit()
    report = store.integrity(build)
    assert report["consistent"] is False
    assert report["chunks_without_vector"] == 1


def test_temporal_filter_is_a_plain_where(store):
    chunks = [
        chunk("old#0", "nk1/st1/p1", valid_from=20200101, valid_to=20220101, article="1", clause="1"),
        chunk("new#0", "nk1/st1/p2", valid_from=20220101, article="1", clause="2"),
    ]
    store.load_chunks(chunks, corpus_build_id(chunks))
    store.connection.commit()

    assert "old#0" in store.candidates(as_of=20210101)
    assert "new#0" not in store.candidates(as_of=20210101)
    assert "new#0" in store.candidates(as_of=20230101)


def test_repealed_chunks_excluded_by_default(store):
    chunks = [chunk("dead#0", "nk1/st25/p1", status="repealed", article="25", clause="1")]
    store.load_chunks(chunks, corpus_build_id(chunks))
    store.connection.commit()
    assert "dead#0" not in store.candidates()
    assert "dead#0" in store.candidates(status=None)


def test_one_hop_traversal_in_one_query(store):
    chunks = [
        chunk("src#0", "nk1/st77/p1", refs=(("nk1/st78/p1", "exception"),), article="77", clause="1"),
        chunk("dst#0", "nk1/st78/p1", article="78", clause="1"),
    ]
    store.load_chunks(chunks, corpus_build_id(chunks))
    store.connection.commit()
    assert store.expand_refs(["src#0"], ("procedure", "exception", "definition")) == ["dst#0"]


def test_untyped_references_not_expanded(store):
    chunks = [
        chunk("s2#0", "nk1/st79/p1", refs=(("nk1/st80/p1", "amendment"),), article="79", clause="1"),
        chunk("d2#0", "nk1/st80/p1", article="80", clause="1"),
    ]
    store.load_chunks(chunks, corpus_build_id(chunks))
    store.connection.commit()
    assert store.expand_refs(["s2#0"], ("procedure", "exception", "definition")) == []


def test_vectors_round_trip(store):
    chunks = [chunk("v#0", "nk1/st90/p1", article="90", clause="1")]
    build = corpus_build_id(chunks)
    store.load_chunks(chunks, build)
    store.load_vectors(["v#0"], [[0.25, -0.5, 0.75]], "enc", build)
    store.connection.commit()
    ids, vectors = store.vectors_of(["v#0"], "enc")
    assert ids == ["v#0"]
    assert [round(x, 6) for x in vectors[0]] == [0.25, -0.5, 0.75]


def test_build_id_is_stable_and_content_sensitive():
    a = [chunk("x#0", "nk1/st1/p1")]
    b = [chunk("x#0", "nk1/st1/p1")]
    b[0]["text_bm25"] = "другой текст"
    assert corpus_build_id(a) == corpus_build_id(list(a))
    assert corpus_build_id(a) != corpus_build_id(b)
