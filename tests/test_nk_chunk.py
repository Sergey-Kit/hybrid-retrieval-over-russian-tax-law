"""Чанкинг НК по спеке §2–3: единица, префикс, разделение полей."""
import pytest

from p3.nk.chunk import (
    MAX_TOKENS,
    MIN_TOKENS,
    PREFIX_SEPARATOR,
    SENTINEL_VALID_TO,
    chunk_article,
    estimate_tokens,
)
from p3.nk.parse import Article, Clause


def article(clauses, **kwargs):
    defaults = dict(
        number="88", title="Камеральная налоговая проверка", part=1,
        section="V", section_title="Налоговый контроль",
        chapter="14", chapter_title="Налоговый контроль",
    )
    defaults.update(kwargs)
    return Article(clauses=clauses, **defaults)


def long_text(words: int) -> str:
    return " ".join(["слово"] * words)


# --- единица чанка ----------------------------------------------------

def test_one_clause_one_chunk():
    chunks = chunk_article(article([
        Clause("1", long_text(200)), Clause("2", long_text(200)), Clause("3", long_text(200)),
    ]))
    assert len(chunks) == 3
    assert [c.clause for c in chunks] == ["1", "2", "3"]


def test_short_article_is_one_chunk():
    """Статья короче MIN_TOKENS идёт целиком (спека §2)."""
    chunks = chunk_article(article([Clause("1", "Короткий текст."), Clause("2", "И ещё.")]))
    assert len(chunks) == 1
    assert chunks[0].clause is None


def test_article_without_numbered_clauses_is_one_chunk():
    chunks = chunk_article(article([Clause(None, long_text(300))]))
    assert len(chunks) == 1
    assert chunks[0].norm_unit_id == "nk1/st88"


def test_long_clause_splits_by_subclauses():
    body = "Налоговые органы вправе:\n" + "\n".join(
        f"{i}) {long_text(90)}" for i in range(1, 8)
    )
    chunks = chunk_article(article([Clause("1", body)]))
    assert len(chunks) > 1
    assert [c.subclause for c in chunks] == [str(i) for i in range(1, 8)]


def test_subclause_inherits_clause_intro():
    """Подпункт без вводной фразы пункта — обрывок предложения (спека §2)."""
    body = "Налоговые органы вправе:\n" + "\n".join(f"{i}) {long_text(90)}" for i in range(1, 8))
    chunks = chunk_article(article([Clause("1", body)]))
    assert all(c.text_bm25.startswith("Налоговые органы вправе:") for c in chunks)


def test_long_clause_without_subclauses_splits_by_paragraphs():
    body = "\n".join(long_text(200) for _ in range(6))
    chunks = chunk_article(article([Clause("1", body)]))
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text_bm25) <= MAX_TOKENS for c in chunks)


def test_splittable_content_stays_under_the_threshold():
    body = "\n".join(long_text(150) for _ in range(20))
    chunks = chunk_article(article([Clause("1", body)]))
    assert all(estimate_tokens(c.text_bm25) <= MAX_TOKENS for c in chunks)


def test_single_long_paragraph_is_not_cut():
    """Спека §2 запрещает резать предложение, перечисление от вводной фразы и
    условие от последствия. Абзац без внутренних границ дробить нечем, поэтому
    он остаётся целым и превышает порог — осознанно, а не по недосмотру.
    На части первой НК такого пункта нет: за порогом ноль чанков."""
    (chunk,) = chunk_article(article([Clause("1", long_text(900))]))
    assert estimate_tokens(chunk.text_bm25) > MAX_TOKENS
    assert len(chunk.para_offsets) == 1


# --- префикс и поля ---------------------------------------------------

def test_prefix_only_in_dense_field():
    """В BM25 префикс уничтожает различимость: все чанки статьи получают
    одни и те же высокочастотные токены, IDF падает, ранжирование внутри
    статьи становится случайным (спека §3)."""
    (chunk,) = chunk_article(article([Clause("1", long_text(200))]))
    assert "Камеральная налоговая проверка" in chunk.text_dense
    assert "Камеральная налоговая проверка" not in chunk.text_bm25
    assert chunk.text_bm25 == long_text(200)


def test_prefix_carries_full_hierarchy():
    (chunk,) = chunk_article(article([Clause("3", long_text(200))]))
    prefix = chunk.text_dense.split(PREFIX_SEPARATOR)[0]
    assert "часть первая" in prefix
    assert "Раздел V. Налоговый контроль" in prefix
    assert "Глава 14. Налоговый контроль" in prefix
    assert "Статья 88. Камеральная налоговая проверка" in prefix
    assert "Пункт 3" in prefix


def test_title_field_is_separate():
    (chunk,) = chunk_article(article([Clause("3", long_text(200))]))
    assert chunk.title_bm25 == "Камеральная налоговая проверка Статья 88, пункт 3"


def test_dense_uses_the_spec_separator():
    (chunk,) = chunk_article(article([Clause("1", long_text(200))]))
    assert PREFIX_SEPARATOR in chunk.text_dense


# --- идентификаторы и payload -----------------------------------------

def test_three_level_identifiers():
    (chunk,) = chunk_article(article([Clause("3", long_text(200))]), version_no=7)
    assert chunk.norm_unit_id == "nk1/st88/p3"
    assert chunk.version_id == "nk1/st88/p3@r7"
    assert chunk.chunk_id == "nk1/st88/p3@r7#0"


def test_chunk_ids_unique_within_article():
    body = "\n".join(long_text(200) for _ in range(6))
    chunks = chunk_article(article([Clause("1", body)]), version_no=7)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_repealed_clause_kept_with_status():
    chunks = chunk_article(article([
        Clause("1", long_text(200)),
        Clause("2", long_text(200), is_repealed=True),
    ]))
    assert [c.status for c in chunks] == ["active", "repealed"]


def test_valid_to_uses_sentinel_not_null():
    """Фильтр по диапазону однороднее, чем проверка на null (спека §5)."""
    (chunk,) = chunk_article(article([Clause("1", long_text(200))]))
    assert chunk.valid_to == SENTINEL_VALID_TO


def test_dates_are_not_invented():
    """Из ИПС известна только дата изменяющего акта, а спека требует четыре
    несовпадающие даты. Заполнить их одной значило бы получить правдоподобный,
    но неверный темпоральный фильтр."""
    (chunk,) = chunk_article(article([Clause("1", long_text(200))]), version_no=7)
    assert chunk.valid_from is None
    assert chunk.applies_from is None
    assert chunk.published_at is None
    assert chunk.version_no == 7


def test_refs_out_attached_to_chunk():
    (chunk,) = chunk_article(article([
        Clause("1", "в порядке, установленном статьей 93.1 настоящего Кодекса " + long_text(200))
    ]))
    assert {r["target"] for r in chunk.refs_out} == {"nk1/st93.1"}


def test_para_offsets_map_paragraphs():
    (chunk,) = chunk_article(article([Clause("1", "Первый абзац.\nВторой абзац. " + long_text(200))]))
    assert len(chunk.para_offsets) == 2
    start, end = chunk.para_offsets[0]
    assert chunk.text_bm25[start:end] == "Первый абзац."


def test_payload_is_serialisable():
    (chunk,) = chunk_article(article([Clause("1", long_text(200))]))
    payload = chunk.as_payload()
    for key in ("chunk_id", "norm_unit_id", "version_id", "text_dense", "text_bm25",
                "title_bm25", "refs_out", "valid_to", "authority", "source_type"):
        assert key in payload


def test_thresholds_match_the_spec():
    assert (MAX_TOKENS, MIN_TOKENS) == (700, 120)
