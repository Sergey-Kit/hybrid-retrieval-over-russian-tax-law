"""Сборка silver set: разметка, дедупликация, сплит без утечки."""
from pathlib import Path

from p3.silver.build import (
    QUERY_TYPE_LEXICAL,
    QUERY_TYPE_MIXED,
    QUERY_TYPE_NATURAL,
    build,
    classify_query,
    dedup,
    normalize_for_clustering,
    pair_from_decision,
    split,
    summarize,
    write_jsonl,
)
from p3.sources.nalog_ru import Decision, parse_decisions

FIXTURES = Path(__file__).parent / "fixtures"


def make(decision_id: str, topic: str, labels: tuple[str, ...]) -> Decision:
    return Decision(decision_id=decision_id, topic=topic, article_labels=labels)


# --- разметка ---------------------------------------------------------

def test_official_label_becomes_norm_id():
    pair = pair_from_decision(make("1", "Истребование пояснений", ("Статья 93 НК РФ",)))
    assert pair.gold_article_ids == ("nk1:93",)
    assert pair.query_id == "fns-1"
    assert pair.source == "fns_complaint_decision"


def test_part_two_article_gets_part_two_id():
    pair = pair_from_decision(make("2", "Вычет по НДФЛ", ("Статья 220 НК РФ",)))
    assert pair.gold_article_ids == ("nk2:220",)


def test_multiple_labels_are_kept_in_order():
    pair = pair_from_decision(make("3", "Спор", ("Статья 220 НК РФ", "Статья 214.10 НК РФ")))
    assert pair.gold_article_ids == ("nk2:220", "nk2:214.10")


def test_non_nk_label_is_dropped():
    assert pair_from_decision(make("4", "Спор", ("Статья 52 ГК РФ",))) is None


def test_decision_without_usable_labels_is_dropped():
    assert pair_from_decision(make("5", "Спор", ("Глава 26.2 НК РФ",))) is None


# --- тип запроса ------------------------------------------------------

def test_query_without_citation_is_natural():
    assert classify_query("Определение налоговой базы при продаже недвижимости") == QUERY_TYPE_NATURAL


def test_query_that_is_only_a_citation_is_lexical():
    assert classify_query("Статья 93 НК РФ") == QUERY_TYPE_LEXICAL


def test_query_with_citation_and_context_is_mixed():
    text = "Истребование информации в порядке статьи 93 НК РФ в рамках выездной проверки"
    assert classify_query(text) == QUERY_TYPE_MIXED


# --- дедупликация -----------------------------------------------------

def test_clustering_ignores_citation_and_numbers():
    a = normalize_for_clustering("Истребование пояснений в порядке статьи 93 НК РФ")
    b = normalize_for_clustering("Истребование пояснений в порядке статьи 88 НК РФ")
    assert a == b


def test_dedup_collapses_cluster_and_merges_gold():
    pairs = [
        pair_from_decision(make("1", "Истребование пояснений по статье 93 НК РФ", ("Статья 93 НК РФ",))),
        pair_from_decision(make("2", "Истребование пояснений по статье 88 НК РФ", ("Статья 88 НК РФ",))),
    ]
    (merged,) = dedup(pairs)
    assert set(merged.gold_article_ids) == {"nk1:93", "nk1:88"}


def test_dedup_keeps_distinct_topics():
    pairs = [
        pair_from_decision(make("1", "Истребование пояснений", ("Статья 93 НК РФ",))),
        pair_from_decision(make("2", "Вычет по налогу на доходы", ("Статья 220 НК РФ",))),
    ]
    assert len(dedup(pairs)) == 2


# --- сплит ------------------------------------------------------------

def build_pairs(count: int):
    return dedup([
        pair_from_decision(make(str(i), f"Спор о предмете номер {'слово ' * (i % 7)}{i}", ("Статья 93 НК РФ",)))
        for i in range(count)
    ])


def test_split_has_no_cluster_leakage():
    """Главный тест набора: один и тот же спор не может быть и в train, и в test."""
    splits = split(build_pairs(60), seed=42)
    clusters = {name: {p.cluster_id for p in pairs} for name, pairs in splits.items()}
    assert not clusters["train"] & clusters["test"]
    assert not clusters["train"] & clusters["dev"]
    assert not clusters["dev"] & clusters["test"]


def test_split_is_deterministic():
    pairs = build_pairs(40)
    first = {k: [p.query_id for p in v] for k, v in split(pairs, seed=42).items()}
    second = {k: [p.query_id for p in v] for k, v in split(pairs, seed=42).items()}
    assert first == second


def test_split_changes_with_seed():
    pairs = build_pairs(40)
    a = [p.query_id for p in split(pairs, seed=1)["test"]]
    b = [p.query_id for p in split(pairs, seed=2)["test"]]
    assert a != b


def test_split_keeps_every_pair():
    pairs = build_pairs(37)
    splits = split(pairs, seed=7)
    assert sum(len(v) for v in splits.values()) == len(pairs)


# --- сквозной путь ----------------------------------------------------

def test_build_from_real_fixture():
    decisions = parse_decisions((FIXTURES / "fns_decisions.html").read_text(encoding="utf-8"))
    splits = build(decisions)
    report = summarize(splits)
    assert report["total"] == 3
    assert report["distinct_articles"] == 3


def test_write_jsonl_is_stable(tmp_path):
    """Воспроизводимость: второй прогон обязан дать тот же файл побайтово."""
    pairs = build_pairs(12)
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(pairs, first)
    write_jsonl(list(reversed(pairs)), second)
    assert first.read_bytes() == second.read_bytes()
