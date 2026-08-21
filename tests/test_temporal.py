"""Темпоральная модель: версии, интервалы, фильтр, синтетический as_of-тест."""
from datetime import date

import numpy as np
import pytest

from p3.nk.temporal import (
    SENTINEL_VALID_TO, EditionSnapshot, apply_temporal_fields, build_versions,
    changed_norms, clause_unit_of, deduplicate, parse_ru_date, summarize,
    text_hash, to_int_date, version_at,
)
from p3.search.dense import DenseIndex
from p3.silver.temporal_set import build_temporal_cases, temporal_accuracy


def snapshot(rdk, day, texts):
    return EditionSnapshot(rdk=rdk, edition_date=parse_ru_date(day), texts=texts)


# --- версии и интервалы ------------------------------------------------

def test_unchanged_text_stays_one_version():
    versions = build_versions([
        snapshot(1, "01.01.2020", {"nk1/st88/p3": "текст"}),
        snapshot(2, "01.01.2022", {"nk1/st88/p3": "текст"}),
    ])
    (only,) = versions["nk1/st88/p3"]
    assert only.editions == [1, 2]
    assert only.valid_to == SENTINEL_VALID_TO


def test_changed_text_closes_the_previous_interval():
    versions = build_versions([
        snapshot(1, "01.01.2020", {"nk1/st88/p3": "старый"}),
        snapshot(2, "01.01.2022", {"nk1/st88/p3": "новый"}),
    ])
    first, second = versions["nk1/st88/p3"]
    assert (first.valid_from, first.valid_to) == (20200101, 20220101)
    assert (second.valid_from, second.valid_to) == (20220101, SENTINEL_VALID_TO)


def test_snapshots_ordered_by_date_not_by_rdk():
    """rdk в ИПС не хронологичен: редакция 212 от 2020 года, 197 — от 2022."""
    versions = build_versions([
        snapshot(212, "29.12.2020", {"nk1/st88/p3": "ранний"}),
        snapshot(197, "28.12.2022", {"nk1/st88/p3": "поздний"}),
    ])
    history = versions["nk1/st88/p3"]
    assert history[0].text == "ранний"
    assert history[0].valid_from < history[1].valid_from


def test_whitespace_does_not_create_a_version():
    assert text_hash("а  б") == text_hash("а б\n")


def test_version_at_date():
    versions = build_versions([
        snapshot(1, "01.01.2020", {"n": "a"}),
        snapshot(2, "01.01.2022", {"n": "b"}),
    ])["n"]
    assert version_at(versions, 20210101).text == "a"
    assert version_at(versions, 20230101).text == "b"
    assert version_at(versions, 20190101) is None


def test_changed_norms_selects_multi_version_only():
    versions = build_versions([
        snapshot(1, "01.01.2020", {"stable": "x", "moving": "a"}),
        snapshot(2, "01.01.2022", {"stable": "x", "moving": "b"}),
    ])
    assert set(changed_norms(versions)) == {"moving"}
    assert summarize(versions)["norm_units_changed"] == 1


def test_dates_are_integers_not_timestamps():
    assert to_int_date(date(2026, 4, 17)) == 20260417


# --- привязка чанков ---------------------------------------------------

def test_clause_unit_ignores_subclause_suffix():
    chunk = {"part": 1, "article": "31", "clause": "1", "norm_unit_id": "nk1/st31/p1/sp2"}
    assert clause_unit_of(chunk) == "nk1/st31/p1"


def test_temporal_fields_applied_by_edition_membership():
    versions = build_versions([
        snapshot(1, "01.01.2020", {"nk1/st88/p3": "старый"}),
        snapshot(2, "01.01.2022", {"nk1/st88/p3": "новый"}),
    ])
    chunk = {
        "part": 1, "article": "88", "clause": "3",
        "norm_unit_id": "nk1/st88/p3", "chunk_id": "nk1/st88/p3@r2#0",
    }
    (updated,) = apply_temporal_fields([chunk], versions, rdk=2)
    assert updated["valid_from"] == 20220101
    assert updated["version_id"] == "nk1/st88/p3@v2"
    assert updated["chunk_id"] == "nk1/st88/p3@v2#0"


def test_deduplicate_collapses_repeated_versions():
    chunks = [{"chunk_id": "a#0"}, {"chunk_id": "a#0"}, {"chunk_id": "b#0"}]
    assert len(deduplicate(chunks)) == 2


# --- фильтр по дате ----------------------------------------------------

def test_dense_filter_applies_before_search():
    """Фильтровать после поиска нельзя: отсечённые кандидаты съедают top-k."""
    index = DenseIndex(
        doc_ids=["old", "new"],
        vectors=np.array([[1.0, 0.0], [0.99, 0.14]], dtype=np.float32),
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert index.search(query, top_k=1)[0][0] == "old"
    assert index.search(query, top_k=1, allowed={"new"})[0][0] == "new"


def test_dense_filter_excluding_everything_returns_nothing():
    index = DenseIndex(doc_ids=["a"], vectors=np.array([[1.0, 0.0]], dtype=np.float32))
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), allowed=set()) == []


def test_bm25_filter():
    from p3.search.bm25 import build_index
    chunks = [
        {"chunk_id": "old", "norm_unit_id": "nk1/st88/p3", "text_bm25": "камеральная проверка", "title_bm25": ""},
        {"chunk_id": "new", "norm_unit_id": "nk1/st88/p3", "text_bm25": "камеральная проверка", "title_bm25": ""},
    ]
    index = build_index(chunks)
    assert {d for d, _ in index.search("камеральная проверка", allowed={"new"})} == {"new"}


# --- синтетический as_of-набор -----------------------------------------

def versions_for(unit="nk1/st93/p3"):
    return build_versions([
        snapshot(1, "01.01.2020", {unit: "старый текст"}),
        snapshot(2, "01.01.2022", {unit: "новый текст"}),
    ])


def test_cases_built_only_for_changed_norms():
    cases = build_temporal_cases(versions_for())
    assert cases, "у изменявшейся нормы должны быть запросы"
    assert all(c.pair.source == "synthetic_as_of" for c in cases)


def test_no_cases_when_nothing_changed():
    versions = build_versions([snapshot(1, "01.01.2020", {"nk1/st93/p3": "текст"})])
    assert build_temporal_cases(versions) == []


def test_probe_date_falls_inside_the_interval():
    for case in build_temporal_cases(versions_for()):
        history = versions_for()[case.pair.source_id.split("@")[0]]
        version = next(v for v in history if v.version_id == case.expected_version_id)
        assert version.valid_from <= case.as_of < version.valid_to


def test_query_carries_the_date():
    for case in build_temporal_cases(versions_for()):
        assert "редакции на" in case.pair.query or "по состоянию на" in case.pair.query


def test_gold_is_a_version_not_just_a_norm():
    for case in build_temporal_cases(versions_for()):
        assert "@v" in case.pair.gold_norm_ids[0]


def test_accuracy_separates_wrong_version_from_missing_norm():
    """Разница принципиальная: чужая редакция выглядит правильным ответом."""
    cases = build_temporal_cases(versions_for())
    case = cases[0]
    unit = case.expected_version_id.split("@")[0]
    other = f"{unit}@v9"

    assert temporal_accuracy([case], {case.pair.query_id: [case.expected_version_id]})[
        "correct_version"] == 1
    assert temporal_accuracy([case], {case.pair.query_id: [other]})[
        "wrong_version_of_right_norm"] == 1
    assert temporal_accuracy([case], {case.pair.query_id: ["nk1/st1/p1@v1"]})[
        "norm_not_found"] == 1
