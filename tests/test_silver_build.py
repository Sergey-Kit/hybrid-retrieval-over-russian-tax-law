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
    assert pair.gold_article_ids == ("nk1/st93",)
    assert pair.query_id == "fns-1"
    assert pair.source == "fns_complaint_decision"


def test_part_two_article_gets_part_two_id():
    pair = pair_from_decision(make("2", "Вычет по НДФЛ", ("Статья 220 НК РФ",)))
    assert pair.gold_article_ids == ("nk2/st220",)


def test_multiple_labels_are_kept_in_order():
    pair = pair_from_decision(make("3", "Спор", ("Статья 220 НК РФ", "Статья 214.10 НК РФ")))
    assert pair.gold_article_ids == ("nk2/st220", "nk2/st214.10")


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
    assert set(merged.gold_article_ids) == {"nk1/st93", "nk1/st88"}


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


# --- письма как второй источник ---------------------------------------

def make_letter(letter_id: str, question: str, labels: tuple[str, ...]):
    from p3.sources.fns_letters import Letter
    return Letter(letter_id=letter_id, question=question, article_labels=labels)


def test_pair_from_letter_uses_official_tags():
    from p3.silver.build import SOURCE_LETTER, pair_from_letter
    pair = pair_from_letter(make_letter("9", "О сроке ответа на требование", ("Статья 93 НК РФ",)))
    assert pair.gold_article_ids == ("nk1/st93",)
    assert pair.source == SOURCE_LETTER
    assert pair.query_id == "let-9"


def test_administrative_letter_is_not_a_pair():
    """Эталон есть, а запроса по существу нет — такая пара только шумит."""
    from p3.silver.build import pair_from_letter
    assert pair_from_letter(
        make_letter("10", "Об утверждении приказа ФНС России", ("Статья 93 НК РФ",))
    ) is None


def test_letter_without_nk_tags_is_dropped():
    from p3.silver.build import pair_from_letter
    assert pair_from_letter(make_letter("11", "Спор", ("Статья 52 ГК РФ",))) is None


def test_build_merges_both_sources():
    from p3.silver.build import SOURCE_DECISION, SOURCE_LETTER
    decisions = [make("1", "Истребование документов у контрагента", ("Статья 93.1 НК РФ",))]
    letters = [make_letter("2", "О порядке заверения копий документов", ("Статья 93 НК РФ",))]
    splits = build(decisions, letters)
    sources = {p.source for pairs in splits.values() for p in pairs}
    assert sources == {SOURCE_DECISION, SOURCE_LETTER}


def test_identical_topic_from_both_sources_lands_in_one_split():
    """Иначе получится утечка между источниками, незаметная в метриках."""
    decisions = [make("1", "О сроке ответа на требование", ("Статья 93 НК РФ",))]
    letters = [make_letter("2", "О сроке ответа на требование", ("Статья 93.1 НК РФ",))]
    splits = build(decisions, letters)
    non_empty = [name for name, pairs in splits.items() if pairs]
    assert len(non_empty) == 1


def test_summarize_reports_layer1_and_part1():
    decisions = [
        make("1", "Истребование документов в порядке камеральной проверки", ("Статья 88 НК РФ",)),
        make("2", "Вычет по налогу на доходы физических лиц", ("Статья 220 НК РФ",)),
    ]
    report = summarize(build(decisions))
    assert report["layer1_pairs"] == 1
    assert report["part1_pairs"] == 1
    assert report["by_source"]["fns_complaint_decision"] == 2
