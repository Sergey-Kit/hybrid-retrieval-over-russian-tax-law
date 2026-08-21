"""BM25F: полевые веса, нормировка длины по полям, поиск по ссылке."""
from p3.search.bm25 import DEFAULT_FIELD_WEIGHTS, Bm25fIndex, build_index


def chunk(chunk_id, norm_unit_id, body, title=""):
    return {
        "chunk_id": chunk_id,
        "norm_unit_id": norm_unit_id,
        "text_bm25": body,
        "title_bm25": title,
    }


def test_finds_norm_by_its_own_citation():
    """Тело пункта ссылается на соседей, но не на себя — норму надо находить
    по её же цитате, и идентичность чанка не должна зависеть от регекспа."""
    index = build_index([
        chunk("a", "nk1/st88/p3", "Камеральной проверкой выявлены ошибки в декларации."),
        chunk("b", "nk1/st93/p1", "Должностное лицо вправе истребовать документы."),
    ])
    assert index.search("п. 3 ст. 88 НК РФ")[0][0] == "a"


def test_article_citation_matches_all_its_clauses():
    index = build_index([
        chunk("a", "nk1/st88/p1", "Первый пункт."),
        chunk("b", "nk1/st88/p3", "Третий пункт."),
        chunk("c", "nk1/st93/p1", "Чужая статья."),
    ])
    found = {doc for doc, _ in index.search("статья 88 НК РФ")}
    assert {"a", "b"} <= found


def test_per_field_length_normalisation():
    """При общей нормировке длинное тело топит короткое поле с сильным сигналом:
    точная ссылка проигрывала частотным словам в коротком документе."""
    long_body = " ".join(["слово"] * 400)
    index = build_index([
        chunk("target", "nk1/st88/p3", long_body),
        chunk("noise", "nk1/st99/p1", "рф рф рф"),
    ])
    assert index.search("п. 3 ст. 88 НК РФ")[0][0] == "target"


def test_title_field_contributes():
    index = build_index([
        chunk("a", "nk1/st1/p1", "нейтральный текст", title="камеральная проверка"),
        chunk("b", "nk1/st2/p1", "нейтральный текст", title="уплата налога"),
    ])
    assert index.search("камеральная проверка")[0][0] == "a"


def test_field_weights_are_configurable():
    chunks = [chunk("a", "nk1/st1/p1", "тело", title="особое слово")]
    with_title = build_index(chunks, field_weights={"body": 1.0, "title": 1.0, "ref": 1.0})
    without_title = build_index(chunks, field_weights={"body": 1.0, "title": 0.0, "ref": 1.0})
    assert with_title.search("особое слово")
    assert not without_title.search("особое слово")


def test_default_weights_follow_the_spec():
    assert DEFAULT_FIELD_WEIGHTS["body"] == 1.0
    assert DEFAULT_FIELD_WEIGHTS["title"] == 0.3


def test_unknown_term_returns_nothing():
    index = build_index([chunk("a", "nk1/st1/p1", "текст нормы")])
    assert index.search("абсолютнонесуществующееслово") == []


def test_empty_index_is_safe():
    index = Bm25fIndex()
    index.finalize()
    assert index.search("что угодно") == []


def test_results_are_deterministic():
    chunks = [chunk(f"c{i}", f"nk1/st{i}/p1", "одинаковый текст нормы") for i in range(5)]
    index = build_index(chunks)
    assert index.search("текст нормы") == index.search("текст нормы")


def test_top_k_respected():
    chunks = [chunk(f"c{i}", f"nk1/st{i}/p1", "текст нормы") for i in range(10)]
    index = build_index(chunks)
    assert len(index.search("текст нормы", top_k=3)) == 3


def test_index_cache_roundtrip(tmp_path):
    """Сборка на объединённом корпусе занимает десятки минут — без кэша каждый
    прогон оценки платил бы это заново."""
    chunks = [chunk("a", "nk1/st88/p3", "камеральная проверка")]
    first = build_index(chunks, cache_dir=tmp_path)
    cached = list(tmp_path.glob("bm25-*.pkl"))
    assert len(cached) == 1

    second = build_index(chunks, cache_dir=tmp_path)
    assert second.doc_ids == first.doc_ids
    assert second.search("камеральная проверка") == first.search("камеральная проверка")


def test_cache_key_changes_with_corpus(tmp_path):
    build_index([chunk("a", "nk1/st1/p1", "текст")], cache_dir=tmp_path)
    build_index([chunk("a", "nk1/st1/p1", "текст"), chunk("b", "nk1/st2/p1", "другой")],
                cache_dir=tmp_path)
    assert len(list(tmp_path.glob("bm25-*.pkl"))) == 2


def test_cache_key_changes_with_weights(tmp_path):
    chunks = [chunk("a", "nk1/st1/p1", "текст", title="заголовок")]
    build_index(chunks, cache_dir=tmp_path, field_weights={"body": 1.0, "title": 0.3, "ref": 2.0})
    build_index(chunks, cache_dir=tmp_path, field_weights={"body": 1.0, "title": 1.0, "ref": 2.0})
    assert len(list(tmp_path.glob("bm25-*.pkl"))) == 2
